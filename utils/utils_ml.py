import torch
from tqdm import tqdm
import torch.nn.functional as trchfnctnl

def compute_accuracy(predictions: torch.tensor, targets: torch.tensor):
    """
    compute the model's accuracy
    :param predictions: tensor containing the predicted labels
    :param targets: tensor containing the target labels
    :return: the accuracy in [0, 1]
    """
    accuracy = torch.div(torch.sum(predictions == targets), len(targets))
    return accuracy


def compute_logits_return_labels_and_predictions(model, dataloader, device=torch.device("cpu"), *args, **kwargs):
    """
    compute the logits given input data loader and model
    :param model: model utilized for the logits computation
    :param dataloader: loader for the training data
    :param device: device used for computation
    :return: logits and targets
    """
    logits = []
    labels = []
    predictions = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, ascii=True, ncols=50, colour='red')):
            # print('m', data[0])
            # exit()
            preds_logit = model(data.to(device))
            logits.append(preds_logit.detach().cpu())

            soft_prob = trchfnctnl.softmax(preds_logit, dim=1)
            if 'print_sp' in kwargs:
                if kwargs['print_sp']:
                    print(soft_prob)
            preds = torch.argmax(soft_prob, dim=1)
            predictions.append(preds.detach().cpu().reshape(-1, 1))

            labels.append(target.detach().cpu().reshape(-1, 1))

    logits = torch.vstack(logits)
    labels = torch.vstack(labels).reshape(-1)
    predictions = torch.vstack(predictions).reshape(-1)
    return logits, labels, predictions
