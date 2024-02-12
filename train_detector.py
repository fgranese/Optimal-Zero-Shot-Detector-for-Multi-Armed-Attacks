import torch
import numpy as np
from tqdm import tqdm
from os.path import join
from losses.losses_detector import detector_loss
from sklearn.metrics import accuracy_score
import logging
from utils.utils_models import load_detectors, create_detectors
from utils import utils_general


# ----------------------- TRAIN -----------------------------

def train_detector(args, classifier, train_loader, device, logger, directory, loss_train):
    # ---- Create detectors ----
    if args.DETECTOR.resume_train_detector:
        detectors, optimizers = load_detectors(args, model=classifier, device=device)
        initial_epoch = args.DETECTOR.resume_epoch + 1
    else:
        detectors, optimizers = create_detectors(args, model=classifier, device=device)
        initial_epoch = 1

    path_detectors = join(args.DETECTOR.detector_dir, args.DATA_NATURAL.data_name)
    utils_general.prep_folder(path=path_detectors, to_file=False)

    detector_id = -1

    print("\nSingle train:", loss_train)
    for epoch in range(initial_epoch, args.DETECTOR.epochs_detector + 1):
        _ = single_detector_train(args,
                                  loss_train=loss_train,
                                  classifier=classifier,
                                  detector=detectors[detector_id],
                                  detector_id=detector_id,
                                  device=device,
                                  train_loader=train_loader,
                                  optimizer=optimizers[detector_id],
                                  epoch=epoch,
                                  directory=directory)

    return detectors


def single_detector_train(args, loss_train, classifier, detector, detector_id, device, train_loader, optimizer, epoch, directory):
    detector.train()

    bar = tqdm(enumerate(train_loader), ascii=True, ncols=70, total=len(train_loader), colour='green')
    acc_epoch = 0
    loss_epoch = 0

    nat = []
    adv = []

    for batch_idx, (data, target) in bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        for epsilon in [args.TRAIN.PGDi.epsilon]:
            detector.eval()

            step_size = epsilon / 8
            loss, y_pred, y_true = detector_loss(loss_train=loss_train,
                                                 classifier=classifier,
                                                 detector=detector,
                                                 detector_id=detector_id,
                                                 x_natural=data,
                                                 y=target,
                                                 device=device,
                                                 step_size=step_size,
                                                 epsilon=epsilon,
                                                 perturb_steps=args.TRAIN.PGDi.perturb_steps)
            detector.train()

            loss.backward()
            optimizer.step()

            acc_epoch += accuracy_score(y_true=y_true.detach().cpu(), y_pred=torch.round(y_pred).detach().cpu())
            loss_epoch += loss.item()

            nat += y_pred[: args.DETECTOR.batch_size].T.tolist()[0]
            adv += y_pred[args.DETECTOR.batch_size:].T.tolist()[0]

    print('Naturals :', np.mean(nat), np.std(nat))
    print('Adversarial :', np.mean(adv), np.std(adv))

    logging.info('Train Epoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, loss_epoch / len(train_loader), acc_epoch / len(train_loader)))

    torch.save({'epoch': epoch,
                'model_state_dict': detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, '{}/detector_epoch_{}.pt'.format(directory, epoch))

    return detector
