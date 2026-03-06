import logging
from model import *
from utils import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def compute_model_averaging_accuracy(models, weights, train_dl, test_dl, n_classes, args):
    """An variant of fedaveraging"""
    avg_cnn = ModerateCNN()

    new_state_dict = {}

    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(avg_cnn.state_dict().items()):
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

    avg_cnn.load_state_dict(new_state_dict)

    # switch to eval mode:
    avg_cnn.eval()

    correct, total = 0, 0
    for batch_idx, (x, target) in enumerate(test_dl):
        out_k = avg_cnn(x)
        _, pred_label = torch.max(out_k, 1)
        total += x.data.size()[0]
        correct += (pred_label == target.data).sum().item()

    logger.info("Accuracy for Fed Averaging correct: {}, total: {}".format(correct, total))


def compute_full_cnn_accuracy(models, weights, train_dl, test_dl, n_classes, device, args):
    num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0],
                   weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
    input_dim = weights[12].shape[0]
    hidden_dims = [weights[12].shape[1], weights[14].shape[1]]

    matched_cnn = ModerateCNNContainer(3, num_filters, kernel_size=3,
                                       input_dim=input_dim, hidden_dims=hidden_dims, output_dim=10)

    new_state_dict = {}
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

    matched_cnn.to(device)
    matched_cnn.eval()

    correct, total = 0, 0
    for batch_idx, (x, target) in enumerate(test_dl):
        x, target = x.to(device), target.to(device)
        out_k = matched_cnn(x)
        _, pred_label = torch.max(out_k, 1)
        total += x.data.size()[0]
        correct += (pred_label == target.data).sum().item()

    logger.info("Accuracy for Neural Matching correct: {}, total: {}".format(correct, total))



