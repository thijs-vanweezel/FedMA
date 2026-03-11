from utils import *
import pickle
import copy
from sklearn.preprocessing import normalize

from matching.pfnm import layer_wise_group_descent
from matching.pfnm import block_patching, patch_weights

from matching_performance import compute_model_averaging_accuracy, compute_full_cnn_accuracy
from tqdm.auto import tqdm
import multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

args_logdir = "logs/imagenet"
#args_dataset = "cifar10"
args_datadir = "/thesis/imnetproc"
args_init_seed = 0
args_net_config = [3072, 100, 10]
args_experiment = ["u-ensemble", "pdm"]
args_trials = 1
#args_lr = 0.01
args_epochs = 5
args_reg = 1e-5
args_alpha = 1.
args_communication_rounds = 5
args_iter_epochs=None

args_pdm_sig = 1.0
args_pdm_sig0 = 1.0
args_pdm_gamma = 1.0


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet', metavar='N',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='imagenet', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--partition', type=str, default='hetero-dir', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--retrain_lr', type=float, default=1e-3, metavar='RLR',
                        help='learning rate using in specific for local network retrain (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained in a training process')
    parser.add_argument('--retrain_epochs', type=int, default=5, metavar='REP',
                        help='how many epochs will be trained in during the locally retraining process')
    parser.add_argument('--n_nets', type=int, default=4, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--retrain', type=bool, default=True, 
                            help='whether to retrain the model or load model locally')
    parser.add_argument('--comm_type', type=str, default='layerwise', 
                            help='which type of communication strategy is going to be used: layerwise/blockwise')    
    parser.add_argument('--comm_round', type=int, default=2, 
                            help='how many round of communications we shoud use')  
    parser.add_argument('--datadir', type=str, default='/thesis/imnetproc',
                            help='path to dataset directory')
    args = parser.parse_args()
    return args

def trans_next_conv_layer_forward(layer_weight, next_layer_shape):
    reshaped = layer_weight.reshape(next_layer_shape).transpose((1, 0, 2, 3)).reshape((next_layer_shape[1], -1))
    return reshaped

def trans_next_conv_layer_backward(layer_weight, next_layer_shape):
    reconstructed_next_layer_shape = (next_layer_shape[1], next_layer_shape[0], next_layer_shape[2], next_layer_shape[3])
    reshaped = layer_weight.reshape(reconstructed_next_layer_shape).transpose(1, 0, 2, 3).reshape(next_layer_shape[0], -1)
    return reshaped

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    losses, running_losses = [], []

    for epoch in tqdm(range(epochs)): # TODO: add early stopping based on validation set
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(tqdm(train_dataloader, leave=False)):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def local_train(nets, args, net_dataidx_map, device="cpu"):
    # save local dataset
    local_datasets = []
    for net_id, net in nets.items():
        if args.retrain:
            dataidxs = net_dataidx_map[net_id]
            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
            # move the model to cuda device:
            net.to(device)

            train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 64, dataidxs, n_clients=args.n_nets,
                                                     num_workers=6, multiprocessing_context=mp.get_context("spawn"), persistent_workers=True)
            train_dl_global, test_dl_global = get_dataloader(args.dataset, args_datadir, args.batch_size, 64,
                                                     num_workers=6, multiprocessing_context=mp.get_context("spawn"), persistent_workers=True)

            local_datasets.append((train_dl_local, test_dl_local)) # TODO: why

            # switch to global test set here
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_global, args.epochs, args.lr, args, device=device)
            # saving the trained models here
            save_model(net, net_id)
        else:
            load_model(net, net_id, device=device)

    nets_list = list(nets.values())
    return nets_list


def local_retrain(local_datasets, weights, args, mode="bottom-up", freezing_index=0, ori_assignments=None, device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freez the entire network
    """

    # ------------------------------------------------------------------
    # 1.  Build matched_cnn from the (post-matching) weight list
    # ------------------------------------------------------------------
    if args.model == 'resnet':
        _DEFAULT_LAYERS = (2, 2, 2, 2)
        n_main_convs = 1 + 2 * sum(_DEFAULT_LAYERS)   # 33 for ResNet-34
        num_filters   = [weights[2 * i].shape[0] for i in range(n_main_convs)]
        dim_out       = weights[-1].shape[0]           # FC bias shape
        matched_cnn   = ResNetContainer(num_filters, layers=_DEFAULT_LAYERS,
                                        channels_in=3, dim_out=dim_out)

        # Load conv weights and BN biases (β); BN γ left at default 1.
        # FC weight is stored transposed in the weights list (row = input neuron).
        matched_cnn.conv.weight.data = torch.from_numpy(
            weights[0].reshape(matched_cnn.conv.weight.shape)).float()
        matched_cnn.bn.bias.data = torch.from_numpy(weights[1]).float()

        w_idx = 1
        for block in matched_cnn.layers:
            block.conv1.weight.data = torch.from_numpy(
                weights[2 * w_idx].reshape(block.conv1.weight.shape)).float()
            block.bn1.bias.data = torch.from_numpy(weights[2 * w_idx + 1]).float()
            w_idx += 1
            block.conv2.weight.data = torch.from_numpy(
                weights[2 * w_idx].reshape(block.conv2.weight.shape)).float()
            block.bn2.bias.data = torch.from_numpy(weights[2 * w_idx + 1]).float()
            w_idx += 1

        matched_cnn.fc.weight.data = torch.from_numpy(weights[2 * w_idx].T).float()
        matched_cnn.fc.bias.data   = torch.from_numpy(weights[2 * w_idx + 1]).float()

        # ------------------------------------------------------------------
        # 2.  Freeze layers (bottom-up).
        #     n_logical_freeze = freezing_index // 2 logical layers are frozen
        #     (each logical layer = one conv+BN pair, 2 entries in weights list).
        #     Layer groups in logical order:
        #       group 0:  stem conv + bn
        #       groups 1,2:  block-0 (conv1+bn1, conv2+bn2)
        #       groups 3,4:  block-1  ...
        # ------------------------------------------------------------------
        n_logical_freeze = freezing_index // 2

        if n_logical_freeze > 0:
            for p in list(matched_cnn.conv.parameters()) + list(matched_cnn.bn.parameters()):
                p.requires_grad = False

        cur_logical = 1
        for block in matched_cnn.layers:
            # conv1 layer occupies logical index cur_logical
            if cur_logical < n_logical_freeze:
                for p in list(block.conv1.parameters()) + list(block.bn1.parameters()):
                    p.requires_grad = False
            # conv2 (and id_conv if present) occupies logical index cur_logical + 1
            if cur_logical + 1 < n_logical_freeze:
                for p in list(block.conv2.parameters()) + list(block.bn2.parameters()):
                    p.requires_grad = False
                if block.id_conv is not None:
                    for p in block.id_conv.parameters():
                        p.requires_grad = False
            cur_logical += 2

    else:  # moderate-cnn
        num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
        input_dim = weights[12].shape[0]
        hidden_dims = [weights[12].shape[1], weights[14].shape[1]]

        matched_cnn = ModerateCNNContainer(3,
                                            num_filters,
                                            kernel_size=3,
                                            input_dim=input_dim,
                                            hidden_dims=hidden_dims,
                                            output_dim=10)

        new_state_dict = {}
        n_layers = int(len(weights) / 2)
        __non_loading_indices = []

        # handle the conv layers part which is not changing
        for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}

            new_state_dict.update(temp_dict)
        matched_cnn.load_state_dict(new_state_dict)

        for param_idx, param in enumerate(matched_cnn.parameters()):
            # bottom-up: freeze layers before freezing_index
            if param_idx < freezing_index:
                param.requires_grad = False


    matched_cnn.to(device).train()
    # start training last fc layers:
    train_dl_local = local_datasets[0]
    test_dl_local = local_datasets[1]

    if freezing_index < (len(weights) - 2):
        optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=args.retrain_lr)
        retrain_epochs = args.retrain_epochs
    else:
        optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=(args.retrain_lr/10))
        retrain_epochs = int(args.retrain_epochs * 3)

    criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    logger.info('n_training: %d' % len(train_dl_local))
    logger.info('n_test: %d' % len(test_dl_local))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: %f' % train_acc)
    logger.info('>> Pre-Training Test accuracy: %f' % test_acc)

    for epoch in tqdm(range(retrain_epochs)): # TODO: should this also be early stopped based on validation set?
        epoch_loss_collector = []
        for batch_idx, (x, target) in tqdm(enumerate(train_dl_local), leave=False):
            x, target = x.to(device), target.to(device)

            optimizer_fine_tune.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = matched_cnn(x)
            loss = criterion_fine_tune(out, target)
            epoch_loss_collector.append(loss.item())

            loss.backward()
            optimizer_fine_tune.step()

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Epoch Avg Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    return matched_cnn


def reconstruct_local_net(weights, args, ori_assignments=None, worker_index=0):
    if args.model == 'resnet':
        # ---------------------------------------------------------------
        # Reconstruct each worker's local weight list from the globally
        # matched weights by slicing according to the worker's assignments.
        # Format mirrors pdm_prepare_full_weights_cnn for ResNet:
        #   [initial_conv_w, initial_bn_b,
        #    block0_conv1_w, block0_bn1_b, block0_conv2_w, block0_bn2_b,
        #    ..., fc_w_T, fc_b]
        # id_conv weights are not in the matched list; they are re-initialised
        # randomly and trained during local_retrain.
        # ---------------------------------------------------------------
        _DEFAULT_LAYERS = (2, 2, 2, 2)
        n_main_convs = 1 + 2 * sum(_DEFAULT_LAYERS)   # 33 for ResNet-34
        reconstructed_weights = []

        # --- logical layer 0: initial stem conv ---
        # Input (RGB) is not matched; only output filters are sliced.
        cur_ass = ori_assignments[0][worker_index]
        reconstructed_weights.append(weights[0][cur_ass, :])          # conv weight
        reconstructed_weights.append(weights[1][cur_ass])              # bn.bias

        # --- logical layers 1 .. n_main_convs-1: block conv pairs ---
        for k in range(1, n_main_convs):
            cur_ass  = ori_assignments[k][worker_index]
            prev_ass = ori_assignments[k - 1][worker_index]
            w = weights[2 * k]                      # (global_out, global_in * 9)
            global_out = w.shape[0]
            global_in  = weights[2 * (k - 1)].shape[0]   # matched neurons in prev layer

            # Step 1: channel slice (permute input channels by prev_ass)
            ori_shape_global = (global_out, global_in, 3, 3)
            trans_w    = trans_next_conv_layer_forward(w, ori_shape_global)
            sliced_ch  = trans_w[prev_ass, :]
            ori_shape_out = (global_out, len(prev_ass), 3, 3)
            w_ch = trans_next_conv_layer_backward(sliced_ch, ori_shape_out)

            # Step 2: filter slice (permute output channels by cur_ass)
            reconstructed_weights.append(w_ch[cur_ass, :])            # conv weight
            reconstructed_weights.append(weights[2 * k + 1][cur_ass]) # bn.bias

        # --- logical layer n_main_convs: fully-connected ---
        # weights[2*n_main_convs] = fc.weight.T stored as (global_in, dim_out)
        prev_ass = ori_assignments[n_main_convs - 1][worker_index]
        reconstructed_weights.append(weights[2 * n_main_convs][prev_ass, :])
        reconstructed_weights.append(weights[2 * n_main_convs + 1])   # fc.bias unchanged
        return reconstructed_weights

    # -----------------------------------------------------------------------
    # ModerateCNN path
    # -----------------------------------------------------------------------
    #[(35, 27), (35,), (68, 315), (68,), (132, 612), (132,), (132, 1188), (132,),
    #(260, 1188), (260,), (260, 2340), (260,),
    #(4160, 1025), (1025,), (1025, 515), (515,), (515, 10), (10,)]
    matched_cnn = ModerateCNN()

    num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
    # we need to estimate the output shape here:
    shape_estimator = ModerateCNNContainerConvBlocks(num_filters=num_filters)
    dummy_input = torch.rand(1, 3, 32, 32)
    estimated_output = shape_estimator(dummy_input)
    input_dim = estimated_output.view(-1).size()[0]


    def __reconstruct_weights(weight, assignment, layer_ori_shape, matched_num_filters=None, weight_type="conv_weight", slice_dim="filter"):
        if weight_type == "conv_weight":
            if slice_dim == "filter":
                res_weight = weight[assignment, :]
            elif slice_dim == "channel":
                _ori_matched_shape = list(copy.deepcopy(layer_ori_shape))
                _ori_matched_shape[1] = matched_num_filters
                trans_weight = trans_next_conv_layer_forward(weight, _ori_matched_shape)
                sliced_weight = trans_weight[assignment, :]
                res_weight = trans_next_conv_layer_backward(sliced_weight, layer_ori_shape)
        elif weight_type == "bias":
            res_weight = weight[assignment]
        elif weight_type == "first_fc_weight":
            # NOTE: please note that in this case, we pass the `estimated_shape` to `layer_ori_shape`:
            __ori_shape = weight.shape
            res_weight = weight.reshape(matched_num_filters, layer_ori_shape[2]*layer_ori_shape[3]*__ori_shape[1])[assignment, :]
            res_weight = res_weight.reshape((len(assignment)*layer_ori_shape[2]*layer_ori_shape[3], __ori_shape[1]))
        elif weight_type == "fc_weight":
            if slice_dim == "filter":
                res_weight = weight.T[assignment, :]
                #res_weight = res_weight.T
            elif slice_dim == "channel":
                res_weight = weight[assignment, :]
        return res_weight

    reconstructed_weights = []
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        _matched_weight = weights[param_idx]
        if param_idx < 1: # we need to handle the 1st conv layer specificly since the color channels are aligned
            _assignment = ori_assignments[int(param_idx / 2)][worker_index]
            _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=_assignment, 
                                                layer_ori_shape=param.size(), matched_num_filters=None,
                                                weight_type="conv_weight", slice_dim="filter")
            reconstructed_weights.append(_res_weight)

        elif (param_idx >= 1) and (param_idx < len(weights) -2):
            if "bias" in key_name: # the last bias layer is already aligned so we won't need to process it
                _assignment = ori_assignments[int(param_idx / 2)][worker_index]
                _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=_assignment, 
                                        layer_ori_shape=param.size(), matched_num_filters=None,
                                        weight_type="bias", slice_dim=None)
                reconstructed_weights.append(_res_bias)

            elif "conv" in key_name or "features" in key_name:
                # we make a note here that for these weights, we will need to slice in both `filter` and `channel` dimensions
                cur_assignment = ori_assignments[int(param_idx / 2)][worker_index]
                prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
                _matched_num_filters = weights[param_idx - 2].shape[0]
                _layer_ori_shape = list(param.size())
                _layer_ori_shape[0] = _matched_weight.shape[0]

                _temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment, 
                                                    layer_ori_shape=_layer_ori_shape, matched_num_filters=_matched_num_filters,
                                                    weight_type="conv_weight", slice_dim="channel")

                _res_weight = __reconstruct_weights(weight=_temp_res_weight, assignment=cur_assignment, 
                                                    layer_ori_shape=param.size(), matched_num_filters=None,
                                                    weight_type="conv_weight", slice_dim="filter")
                reconstructed_weights.append(_res_weight)

            elif "fc" in key_name or "classifier" in key_name:
                # we make a note here that for these weights, we will need to slice in both `filter` and `channel` dimensions
                cur_assignment = ori_assignments[int(param_idx / 2)][worker_index]
                prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
                _matched_num_filters = weights[param_idx - 2].shape[0]

                if param_idx != 12: # this is the index of the first fc layer
                    #logger.info("%%%%%%%%%%%%%%% prev assignment length: {}, cur assignmnet length: {}".format(len(prev_assignment), len(cur_assignment)))
                    temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="fc_weight", slice_dim="channel")

                    _res_weight = __reconstruct_weights(weight=temp_res_weight, assignment=cur_assignment, 
                                                        layer_ori_shape=param.size(), matched_num_filters=None,
                                                        weight_type="fc_weight", slice_dim="filter")

                    reconstructed_weights.append(_res_weight.T)
                else:
                    # that's for handling the first fc layer that is connected to the conv blocks
                    temp_res_weight = __reconstruct_weights(weight=_matched_weight, assignment=prev_assignment, 
                                                        layer_ori_shape=estimated_output.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="first_fc_weight", slice_dim=None)

                    _res_weight = __reconstruct_weights(weight=temp_res_weight, assignment=cur_assignment, 
                                                        layer_ori_shape=param.size(), matched_num_filters=None,
                                                        weight_type="fc_weight", slice_dim="filter")
                    reconstructed_weights.append(_res_weight.T)
        elif param_idx  == len(weights) - 2:
            # this is to handle the weight of the last layer
            prev_assignment = ori_assignments[int(param_idx / 2)-1][worker_index]
            _res_weight = _matched_weight[prev_assignment, :]
            reconstructed_weights.append(_res_weight)
        elif param_idx  == len(weights) - 1:
            reconstructed_weights.append(_matched_weight)

    return reconstructed_weights


def BBP_MAP(nets_list, model_meta_data, layer_type, net_dataidx_map,
                            averaging_weights, args,
                            n_classes=None,
                            device="cpu"):
    # starting the neural matching
    models = nets_list
    cls_freqs = traindata_cls_counts
    if n_classes is None:
        n_classes = args_net_config[-1]
    it=5
    sigma=args_pdm_sig 
    sigma0=args_pdm_sig0
    gamma=args_pdm_gamma
    assignments_list = []
    
    batch_weights = pdm_prepare_full_weights_cnn(models, device=device)
    raw_batch_weights = copy.deepcopy(batch_weights)
    
    logging.info("=="*15)
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    gamma = 7.0
    sigma = 1.0
    sigma0 = 1.0

    n_layers = int(len(batch_weights[0]) / 2)
    num_workers = len(nets_list)
    matching_shapes = []

    first_fc_index = None

    for layer_index in range(1, n_layers):
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
             batch_weights=batch_weights, 
             layer_index=layer_index,
             sigma0_layers=sigma0, 
             sigma_layers=sigma, 
             batch_frequencies=batch_freqs, 
             it=it, 
             gamma_layers=gamma, 
             model_meta_data=model_meta_data,
             model_layer_type=layer_type,
             n_layers=n_layers,
             matching_shapes=matching_shapes,
             args=args
             )
        assignments_list.append(assignment)
        
        # iii) load weights to the model and train the whole thing
        type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2]
        if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
            l_type = "conv"
        elif 'fc' in type_of_patched_layer or 'classifier' in type_of_patched_layer:
            l_type = "fc"

        type_of_this_layer = layer_type[2 * layer_index - 2]
        type_of_prev_layer = layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in type_of_this_layer or 'classifier' in type_of_this_layer) and ('conv' in type_of_prev_layer or 'features' in type_of_this_layer))
        
        if first_fc_identifier:
            first_fc_index = layer_index
        
        matching_shapes.append(L_next)
        tempt_weights =  [([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) for w in range(num_workers)]

        # i) permutate the next layer wrt matching result
        for worker_index in range(num_workers):
            if first_fc_index is None:
                if l_type == "conv":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2], 
                                        L_next, assignment[worker_index], 
                                        layer_index+1, model_meta_data,
                                        matching_shapes=matching_shapes, layer_type=l_type,
                                        dataset=args.dataset, network_name=args.model)
                elif l_type == "fc":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, 
                                        L_next, assignment[worker_index], 
                                        layer_index+1, model_meta_data,
                                        matching_shapes=matching_shapes, layer_type=l_type,
                                        dataset=args.dataset, network_name=args.model).T

            elif layer_index >= first_fc_index:
                patched_weight = patch_weights(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, L_next, assignment[worker_index]).T

            tempt_weights[worker_index].append(patched_weight)

        # ii) prepare the whole network weights
        for worker_index in range(num_workers):
            for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
                tempt_weights[worker_index].append(batch_weights[worker_index][lid])

        retrained_nets = []
        for worker_index in range(num_workers):
            dataidxs = net_dataidx_map[worker_index]
            train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs, n_clients=args.n_nets,
                                                     num_workers=6, multiprocessing_context=mp.get_context("spawn"), persistent_workers=True)

            logger.info("Re-training on local worker: {}, starting from layer: {}".format(worker_index, 2 * (layer_index + 1) - 2))
            retrained_cnn = local_retrain((train_dl_local,test_dl_local), tempt_weights[worker_index], args, 
                                            freezing_index=(2 * (layer_index + 1) - 2), device=device)
            retrained_nets.append(retrained_cnn)
        batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=device)

    ## we handle the last layer carefully here ...
    ## averaging the last layer
    matched_weights = []
    num_layers = len(batch_weights[0])

    with open('./matching_weights_cache/matched_layerwise_weights', 'wb') as weights_file:
        pickle.dump(batch_weights, weights_file)

    last_layer_weights_collector = []

    for i in range(num_workers):
        # firstly we combine last layer's weight and bias
        bias_shape = batch_weights[i][-1].shape
        last_layer_bias = batch_weights[i][-1].reshape((1, bias_shape[0]))
        last_layer_weights = np.concatenate((batch_weights[i][-2], last_layer_bias), axis=0)
        
        # the directed normalization doesn't work well, let's try weighted averaging
        last_layer_weights_collector.append(last_layer_weights)

    last_layer_weights_collector = np.array(last_layer_weights_collector)
    
    avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)

    for i in range(n_classes):
        avg_weight_collector = np.zeros(last_layer_weights_collector[0][:, 0].shape, dtype=np.float32)
        for j in range(num_workers):
            avg_weight_collector += averaging_weights[j][i]*last_layer_weights_collector[j][:, i]
        avg_last_layer_weight[:, i] = avg_weight_collector

    #avg_last_layer_weight = np.mean(last_layer_weights_collector, axis=0)
    for i in range(num_layers):
        if i < (num_layers - 2):
            matched_weights.append(batch_weights[0][i])

    matched_weights.append(avg_last_layer_weight[0:-1, :])
    matched_weights.append(avg_last_layer_weight[-1, :])
    return matched_weights, assignments_list


def fedma_comm(batch_weights, model_meta_data, layer_type, net_dataidx_map,
                            averaging_weights, args,
                            train_dl_global,
                            test_dl_global,
                            assignments_list,
                            n_classes=10,
                            comm_round=2,
                            device="cpu"):
    '''
    version 0.0.2
    In this version we achieve layerwise matching with communication in a blockwise style
    i.e. we unfreeze a block of layers (each 3 consecutive layers)---> retrain them ---> and rematch them
    '''
    n_layers = int(len(batch_weights[0]) / 2)
    num_workers = len(batch_weights)

    matching_shapes = []
    first_fc_index = None
    gamma = 5.0
    sigma = 1.0
    sigma0 = 1.0

    cls_freqs = traindata_cls_counts
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    it=5

    for cr in range(comm_round):
        logger.info("Entering communication round: {} ...".format(cr))
        retrained_nets = []
        for worker_index in range(args.n_nets):
            dataidxs = net_dataidx_map[worker_index]
            train_dl_local, test_dl_local = get_dataloader(args.dataset, args_datadir, args.batch_size, 32, dataidxs, n_clients=args.n_nets,
                                                     num_workers=6, multiprocessing_context=mp.get_context("spawn"), persistent_workers=True)

            # for the "squeezing" mode, we pass assignment list wrt this worker to the `local_retrain` function
            recons_local_net = reconstruct_local_net(batch_weights[worker_index], args, ori_assignments=assignments_list, worker_index=worker_index)
            retrained_cnn = local_retrain((train_dl_local,test_dl_local), recons_local_net, args,
                                            mode="bottom-up", freezing_index=0, ori_assignments=None, device=device)
            retrained_nets.append(retrained_cnn)

        # BBP_MAP step
        hungarian_weights, assignments_list = BBP_MAP(retrained_nets, model_meta_data, layer_type, net_dataidx_map, averaging_weights, args, n_classes=n_classes, device=device)

        logger.info("After retraining and rematching for comm. round: {}, we measure the accuracy ...".format(cr))
        _ = compute_full_cnn_accuracy(models,
                                   hungarian_weights,
                                   train_dl_global,
                                   test_dl_global,
                                   n_classes,
                                   device=device,
                                   args=args)
        batch_weights = [copy.deepcopy(hungarian_weights) for _ in range(args.n_nets)]
        del hungarian_weights
        del retrained_nets


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    logger.info(device)
    args = add_fit_args(argparse.ArgumentParser(description='Probabilistic Federated CNN Matching'))

    args_datadir = args.datadir

    seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("Partitioning data")

    y_train, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset, args_datadir, args_logdir,
                                                            args.partition, args.n_nets, args_alpha, args=args)

    n_classes = len(np.unique(y_train))
    averaging_weights = np.zeros((args.n_nets, n_classes), dtype=np.float32)

    for i in range(n_classes):
        total_num_counts = 0
        worker_class_counts = [0] * args.n_nets
        for j in range(args.n_nets):
            if i in traindata_cls_counts[j].keys():
                total_num_counts += traindata_cls_counts[j][i]
                worker_class_counts[j] = traindata_cls_counts[j][i]
            else:
                total_num_counts += 0
                worker_class_counts[j] = 0
        averaging_weights[:, i] = worker_class_counts / total_num_counts

    logger.info("averaging_weights: {}".format(averaging_weights))

    logger.info("Initializing nets")
    nets, model_meta_data, layer_type = init_models(args_net_config, args.n_nets, args)
    logger.info("Retrain? : {}".format(args.retrain))

    ### local training stage
    nets_list = local_train(nets, args, net_dataidx_map, device=device)

    train_dl_global, test_dl_global = get_dataloader(args.dataset, args_datadir, args.batch_size, 32,
                                                     num_workers=6, multiprocessing_context=mp.get_context("spawn"), persistent_workers=True)

    hungarian_weights, assignments_list = BBP_MAP(nets_list, model_meta_data, layer_type, net_dataidx_map, averaging_weights, args, n_classes=n_classes, device=device)

    batch_weights = pdm_prepare_full_weights_cnn(nets_list, device=device)
    total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
    logger.info("Total data points: {}".format(total_data_points))
    logger.info("Freq of FedAvg: {}".format(fed_avg_freqs))

    averaged_weights = []
    num_layers = len(batch_weights[0])
    for i in range(num_layers):
        avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
        averaged_weights.append(avegerated_weight)

    for aw in averaged_weights:
        logger.info(aw.shape)

    models = nets_list

    # FedMA communication rounds
    comm_init_batch_weights = [copy.deepcopy(hungarian_weights) for _ in range(args.n_nets)]

    fedma_comm(comm_init_batch_weights,
                             model_meta_data, layer_type, net_dataidx_map,
                             averaging_weights, args,
                             train_dl_global,
                             test_dl_global,
                             assignments_list,
                             n_classes=n_classes,
                             comm_round=args.comm_round,
                             device=device)

