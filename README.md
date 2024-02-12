# Optimal Zero-Shot Detector for Multi-Armed Attacks
We propose a simple yet effective method to aggregate the decisions based on the soft-probability outputs of multiple trained detectors, possibly provided by a third party.

The features and the adversarial examples have been created by executing this <a href="https://github.com/aldahdooh/detectors_review">code</a>. 

Checkpoints for CIFAR10 and SVHN are available <a href="https://drive.google.com/file/d/1OQl6ZelamNUy-40qeKe5Aw5atTrLzdf6/view?usp=share_link">here</a>.

### Current package structure
```
Package
Optimal Zero-Shot Detector for Multi-Armed Attacks/
├── README.md
├── checkpoints
│   ├── classifiers
│   │   ├── cifar10
│   │   │   └── rn-best.pt
│   │   └── svhn
│   │       └── rn-best.pt
│   └── detectors
├── config
│   └── config.yaml
├── config_reader_utls
│   ├── attrDict.py
│   └── config_reader_utls.py
├── datasets
│   └── dataset.py
├── losses
│   ├── losses_classifier.py
│   └── losses_detector.py
├── main.py
├── models
│   ├── detector.py
│   └── resnet.py
├── optimization
│   └── BA.py
├── pipeline_testing.py
├── pipeline_training.py
├── mixture.py
├── train_detector.py
└── utils
    ├── parser.py
    ├── utils_general.py
    ├── utils_ml.py
    ├── utils_models.py
    └── utils_plot.py
```

#### Usage

To execute:
- Create the enviroment:
```console
foo@bar:~$ conda create --name mixture python==3.8.11
```
- Activate the enviroment:
```console
foo@bar:~$ source activate mixture
```
- Install all the required packages:
```console
(mixture) foo@bar:~$ pip3 install -r requirements.txt
```
- Launch the test from CLI for CIFAR10 or SVHN (see <code>config/config.yaml</code>):
```console
(mixture) foo@bar:~$ python main.py 
```





