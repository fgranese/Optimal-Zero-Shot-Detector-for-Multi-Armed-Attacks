import torch
import torch.nn.functional as F
import numpy as np
from losses.losses_classifier import global_loss


# torch.autograd.set_detect_anomaly(True)  # To check nan


def pgdi(x_natural, device, perturb_steps, classifier, step_size, epsilon, loss_train, y, clamp=(0, 1)):
    x_adv = x_natural.clone().detach().requires_grad_(True).to(device)

    for _ in range(perturb_steps):

        loss = global_loss(loss_name=loss_train, preds=classifier(x_adv), y=y, nat=classifier(x_natural))

        with torch.no_grad():
            gradients = torch.autograd.grad(loss, [x_adv], allow_unused=True)[0]
            x_adv += step_size * torch.sign(gradients.detach())

        x_adv = torch.max(torch.min(x_adv, x_natural + epsilon), x_natural - epsilon)

        x_adv = x_adv.clamp(*clamp)

    return x_adv.detach()


def detector_loss(loss_train, classifier, detector, detector_id, x_natural, y, device, step_size=0.003, epsilon=0.031, perturb_steps=10):

    # generate adversarial example
    s_nat = classifier(x_natural).softmax(1)
    nat_correct = [torch.argmax(s_nat.cpu(), axis=1).numpy() == y.cpu().numpy()]
    x_natural_corrected_classified = x_natural[nat_correct].to(device)
    y_natural_corrected_classified = y[nat_correct].to(device)
    x_adv = pgdi(x_natural_corrected_classified, device, perturb_steps, classifier, step_size, epsilon, loss_train, y_natural_corrected_classified)

    s_adv = classifier(x_adv).softmax(1)
    label_adv = torch.tensor(np.float32([torch.argmax(s_adv.cpu(), axis=1).numpy() != y_natural_corrected_classified.cpu().numpy()])).to(device).view((-1, 1))  # 1 if misclassified 0 otherwise (adversarial examples)

    # amount of adv successful
    adv = len(np.where(label_adv.detach().cpu().numpy() == 1)[0])

    classifier_intermediate_layer_nat = classifier.intermediate_forward(input=x_natural)[detector_id]
    detector_output_nat = detector(classifier_intermediate_layer_nat)
    label_nat = torch.zeros(detector_output_nat.shape).to(device)

    if adv > 0:
        x_adv = x_adv[np.where(label_adv.detach().cpu().numpy() == 1)[0]]
        label_adv = label_adv[np.where(label_adv.detach().cpu().numpy() == 1)[0]]

        classifier_intermediate_layer_adv = classifier.intermediate_forward(input=x_adv)[detector_id]
        detector_output_adv = detector(classifier_intermediate_layer_adv)

        detector_outputs = torch.cat((detector_output_nat, detector_output_adv), axis=0)
        detector_outputs = torch.sigmoid(detector_outputs)
        detector_labels = torch.cat((label_nat, label_adv), axis=0)

        w_nat = adv / x_natural.shape[0]
        w_adv = x_natural.shape[0] / adv

    else:
        detector_outputs = detector_output_nat
        detector_outputs = torch.sigmoid(detector_outputs)
        detector_labels = label_nat

        w_nat = 1.
        w_adv = 1.

    loss_detector = F.binary_cross_entropy(detector_outputs, detector_labels, reduction='none')

    w_tensor = torch.cat((torch.full(size=(x_natural.shape[0],), fill_value=w_nat), torch.full(size=(adv,), fill_value=w_adv))).view(-1, 1).to(device)
    final_loss = torch.mean(loss_detector * w_tensor, dim=0)[0]

    return final_loss, detector_outputs, detector_labels
