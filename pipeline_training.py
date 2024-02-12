import os
import torch
from datetime import datetime
from train_detector import train_detector
import torch.backends.cudnn as cudnn
from datasets.dataset import get_dataloader
from utils.utils_models import load_classifier
from utils.utils_general import logging_info, get_device, from_datetime_to_string

def train(args, loss_train):
    device = get_device(args.DETECTOR.device)

    if args.DETECTOR.seed is not None:
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.DETECTOR.seed)

    # ---- Load dataset ----
    train_loader = get_dataloader(data_name=args.DATA_NATURAL.data_name, train=True, batch_size=args.DETECTOR.batch_size)

    # ---- Load model classifier ----
    classifier = load_classifier(checkpoint_dir=args.DETECTOR.classifier_dir, dataset=args.DATA_NATURAL.data_name, device=device).eval()

    from utils import utils_ml as uml
    logits_train, labels_train, preds_train = uml.compute_logits_return_labels_and_predictions(model=classifier, dataloader=train_loader, device=device)
    print('Training accuracy on natural: ', uml.compute_accuracy(predictions=preds_train, targets=labels_train))

    # ---- Training ----
    directory = os.path.join(args.DETECTOR.detector_dir, '{}/{}_{}/'.format(args.DATA_NATURAL.data_name, loss_train, args.TRAIN.PGDi.epsilon))

    file_log = directory + 'training_{}_{}.log'.format(loss_train, from_datetime_to_string(datetime=datetime.now()))
    logger = logging_info(file_path=file_log, args=args)

    detector = train_detector(args=args, classifier=classifier, train_loader=train_loader, device=device, logger=logger, directory=directory, loss_train=loss_train)

    return detector
