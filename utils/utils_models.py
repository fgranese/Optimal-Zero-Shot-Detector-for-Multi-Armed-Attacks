import torch
from models.detector import Detector
import torch.optim as optim


def create_detectors(args, model, device, **kwargs):
    detectors = []
    optimizers = []

    for layer_size in model.intermediate_size():
        use_cuda = torch.cuda.is_available()

        detector = Detector(input_shape=layer_size, drop=0, nodes=args.DETECTOR.DETECTOR_MODEL.nodes, layers=args.DETECTOR.DETECTOR_MODEL.layers)

        if use_cuda:
            detectors.append(detector.to(device))
        else:
            detectors.append(detector)
        optimizers.append(optim.SGD(detector.parameters(), lr=args.DETECTOR.lr,
                                    momentum=args.DETECTOR.momentum,
                                    weight_decay=args.DETECTOR.weight_decay,
                                    nesterov=args.DETECTOR.nesterov))
    return detectors, optimizers


def load_classifier(dataset, checkpoint_dir, device):
    return load_model(dataset_name=dataset, checkpoints_dir=checkpoint_dir, device=device)


def load_model(dataset_name, checkpoints_dir, device):
    if dataset_name == 'cifar10':
        from models.resnet import ResNet18
        path = '{}{}/rn-best.pt'.format(checkpoints_dir, dataset_name)
        model = ResNet18(num_classes=10)
    elif dataset_name == 'svhn':
        from models.resnet import ResNet18
        path = '{}{}/rn-best.pt'.format(checkpoints_dir, dataset_name)
        model = ResNet18(num_classes=10)
    else:
        exit(dataset_name + " not present")

    if torch.cuda.is_available():
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model = model.to(device)
    else:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    return model


def load_detectors(args, model, device, epsilon=None, loss=None):
    loss_train = loss

    detectors = [Detector(input_shape=layer_size, drop=0, nodes=args.DETECTOR.DETECTOR_MODEL.nodes, layers=args.DETECTOR.DETECTOR_MODEL.layers) for
                 layer_size in model.intermediate_size()]
    optimizers = [optim.SGD(detectors[i].parameters(), lr=args.DETECTOR.lr,
                            momentum=args.DETECTOR.momentum,
                            weight_decay=args.DETECTOR.weight_decay,
                            nesterov=args.DETECTOR.nesterov) for i in range(len(detectors))]

    use_cuda = torch.cuda.is_available()

    epoch = args.DETECTOR.resume_epoch

    for i in range(len(detectors)):
        if use_cuda:
            detectors[i] = detectors[i].to(device)

        p = '{}/{}/{}_-1_{}/detector_epoch_{}.pt'.format(args.DETECTOR.detector_dir, args.DATA_NATURAL.data_name, loss_train, args.TRAIN.PGDi.epsilon if epsilon is None else epsilon, epoch)
        print(p)
        checkpoint = torch.load(p, map_location=device)

        if 'optimizer_state_dict' in checkpoint:
            detectors[i].load_state_dict(checkpoint['model_state_dict'])
            optimizers[i].load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

        else:
            detectors[i].load_state_dict(checkpoint)

    return detectors, optimizers
