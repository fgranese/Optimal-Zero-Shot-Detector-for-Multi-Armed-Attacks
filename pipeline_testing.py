import os
import re
import numpy as np
from datasets.dataset import get_dataloader
from utils import utils_ml as uml
from utils.utils_models import load_classifier
from utils.utils_plot import plot_rocs
from utils.utils_general import get_device, from_numpy_to_dataloader
from config_reader_utls.attrDict import AttrDict


def prepare_simultaneous_attack(epsilon_test, norm):
    norm = str(norm)
    if norm == '1':
        admitted_epsilons = [5, 10, 15, 20, 25, 30, 40]
        assert epsilon_test in admitted_epsilons, "Epsilon must be in {}".format(admitted_epsilons)
        algorithms_list = ['pgd1']

    elif norm == '2':
        admitted_epsilons = [0, 0.01, 0.1, 0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        assert epsilon_test in admitted_epsilons, "Epsilon must be in {}".format(admitted_epsilons)
        if epsilon_test == 0.01:
            algorithms_list = ['cw2']
        elif epsilon_test == 0.1:
            algorithms_list = ['hop']
        elif epsilon_test == 0:
            algorithms_list = ['df']
        else:
            algorithms_list = ['pgd2']

    elif norm == 'inf':
        admitted_epsilons = [0.03125, 0.0625, 0.125, 0.25, 0.3125, 0.5]
        assert epsilon_test in admitted_epsilons, "Epsilon must be in {}".format(admitted_epsilons)
        if epsilon_test in [0.03125, 0.0625, 0.25, 0.5]:
            algorithms_list = ['pgdi', 'fgsm', 'bim']
        elif epsilon_test == 0.125:
            algorithms_list = ['pgdi', 'fgsm', 'bim', 'sa']
        elif epsilon_test == 0.3125:
            algorithms_list = ['pgdi', 'fgsm', 'bim', 'cwi']
        else:
            exit("Epsilon must be in [0.03125, 0.0625, 0.125, 0.25, 0.3125, 0.5]")
    else:
        algorithms_list = ['sta']

    losses = ['CE', 'KL', 'g', 'Rao']
    return list(set(['{}_{}_{}'.format(loss, algo, epsilon_test) if re.match(r'.*(fgsm|pgd|bim)', algo) else '_{}'.format(algo)
            for algo in algorithms_list for loss in losses]))


def collect_decision_by_thresholds(probs, thrs_size):
    thrs = np.linspace(probs.min(), probs.max(), thrs_size)
    decision_by_thr = np.zeros((len(probs), thrs_size))

    # An example is detected as adversarial is the prediction is above the threshold, natural if not
    for i in range(thrs_size):
        thr = thrs[i]

        y_pred = np.where(probs > thr, 1, 0)

        decision_by_thr[:, i] = y_pred

    return decision_by_thr


def evaluate_simultaneous_attack(decision_by_thr_adv, decision_by_thr_nat, correct_all, labels_tot):
    # The sample is considered as true positive iff there is at least one successful adversarial examples (i.e. correct_all > 0) and iff the detector detects all of successful adversarial examples (i.e. decision_by_thr_adv = correct_all).
    tp = np.where((decision_by_thr_adv == correct_all) & (correct_all > 0), 1, 0)
    # The sample is considered as false positive if it was a natural example (i.e. labels_tot==0) and if it detected as adversarial (the decision is above 0).
    fp = np.where((decision_by_thr_adv > 0) & (labels_tot == 0), 1, 0)
    # The sample is considered as false positive if it was a natural example (i.e. labels_tot==0) and if it detected as natural (the natural decision is 0).
    tn = np.where((decision_by_thr_nat == 0) & (labels_tot == 0), 1, 0)
    # The sample is considered as false negative iff here is at least one successful adversarial examples (i.e. correct_all > 0) and if it detected less examples than there is (decision_by_thr_adv < correct_all).
    fn = np.where((decision_by_thr_adv < correct_all) & (correct_all > 0), 1, 0)
    return tp, fp, tn, fn


def test(args: AttrDict):
    device = get_device(args.DETECTOR.device)

    # ---- Load model classifier ----
    classifier = load_classifier(checkpoint_dir=args.DETECTOR.classifier_dir, dataset=args.DATA_NATURAL.data_name, device=device).eval()

    # ---- Load natural samples ----
    test_loader_natural = get_dataloader(data_name=args.DATA_NATURAL.data_name, train=False, batch_size=args.DETECTOR.batch_size)
    logits_test_natural, labels_test_natural, preds_test_natural = uml.compute_logits_return_labels_and_predictions(model=classifier, dataloader=test_loader_natural, device=device)
    intermediate_layer_nat = logits_test_natural.to(device)

    attacks = prepare_simultaneous_attack(epsilon_test=args.TEST.ATTACK.epsilon_test, norm=args.TEST.ATTACK.norm)

    correct = []

    for i in range(len(attacks)):
        attack = attacks[i]

        # ---- Load adversarial samples ----
        path_adv = '{}{}/{}{}.npy'.format(args.TEST.ATTACK.data_dir, args.DATA_NATURAL.data_name, args.DATA_NATURAL.data_name, attack)
        print(path_adv)
        X_test_adv = np.load(path_adv)
        test_loader_adversarial = from_numpy_to_dataloader(X=X_test_adv, y=test_loader_natural.dataset[:][1], batch_size=args.DETECTOR.batch_size)
        logits_test_adversarial, labels_test_adv, preds_test_adv = uml.compute_logits_return_labels_and_predictions(model=classifier, dataloader=test_loader_adversarial, device=device)
        intermediate_layer_adv = logits_test_adversarial.to(device)

        # ---- Load Mixture output ----
        if len(args.TEST.detectors) > 1:
            path_dir_probs = '{}{}/all_{}/'.format(args.TEST.RESULTS.res_dir, args.DATA_NATURAL.data_name, args.TRAIN.PGDi.epsilon)
        elif len(args.TEST.detectors) == 1:
            path_dir_probs = f"results/{args.TEST.detectors[0]}/{args.DATA_NATURAL.data_name}/evaluation/"
        else:
            SystemExit("Empty list for detectors")
        file_probs = '{}/probs_{}_all.npy'.format(path_dir_probs, '{}{}'.format(args.DATA_NATURAL.data_name, attack))

        if  os.path.exists(file_probs):
            if len(args.TEST.detectors) == 1 and args.TEST.detectors[0] == 'nss':
                proba = np.load(file_probs)
                proba = proba[:, 1]
            elif len(args.TEST.detectors) == 1 and args.TEST.detectors[0] == 'fs':
                proba = np.load(file_probs)
                proba = (proba-np.min(proba)) / (np.max(proba)-np.min(proba))
            else:
                proba = np.load(file_probs).reshape(-1)

        else:
            from mixture import mixture_output
            proba = mixture_output(args, attack, classifier, device, intermediate_layer_nat, intermediate_layer_adv, len(X_test_adv)).reshape(-1)

        # ---- Compute whether the adversarial examples successfully fools the target classifier or not, and save the decision ----
        correct.append(np.where(preds_test_adv != labels_test_natural, 1, 0))

        labels = np.concatenate((np.zeros(int(len(proba) / 2), ), np.ones(int(len(proba) / 2), )))

        if i == 0:
            # We reshape the variable that if the noisy sample is successful or not
            ca = np.ones((2 * len(X_test_adv), args.TEST.thrs_size))
            for j in range(args.TEST.thrs_size):
                ca[len(X_test_adv):, j] = np.asarray(correct)
            # We multiply the adversarial decision by the successfulness of the attack to discard the non-adversarial samples.
            decision = collect_decision_by_thresholds(proba, args.TEST.thrs_size)
            decision_by_thr_adv = decision * ca
            decision_by_thr_nat = decision
        else:
            # We gather the decisions for all the attacks
            ca = np.ones((2 * len(X_test_adv), args.TEST.thrs_size))
            for j in range(args.TEST.thrs_size):
                ca[len(X_test_adv):, j] = np.asarray(correct)[i, :]
            decision = collect_decision_by_thresholds(proba, args.TEST.thrs_size)
            decision_by_thr_adv = decision_by_thr_adv + decision * ca
            decision_by_thr_nat = decision_by_thr_nat + decision

    correct = np.transpose(correct)
    # We compute the statistics of the successfulness of the attacks
    mean_success_adv_per_sample = correct.mean(1)
    mean_success_adv = mean_success_adv_per_sample.mean(0)

    print('Avg. Number of Successful Attacks per Natural Sample: ', mean_success_adv * len(attacks))
    print('Total. Number of Successful Attacks per Natural Sample: ', len(attacks))

    # We gather the true label (i.e. 0 if the sample is natural, 1 if it is not).
    labels_tot = (np.ones((args.TEST.thrs_size, len(labels))) * labels).transpose()

    correct = np.concatenate((np.zeros(correct.shape), correct), axis=0)
    correct_all = np.zeros(decision_by_thr_adv.shape)
    # We compute the number of times a natural sample has a successful adversarial examples.
    for i in range(decision_by_thr_adv.shape[1]):
        correct_all[:, i] = correct.sum(1)

    tp, fp, tn, fn = evaluate_simultaneous_attack(decision_by_thr_adv, decision_by_thr_nat, correct_all, labels_tot)

    # We sum over all the examples.
    tpr = tp.sum(axis=0) / (tp.sum(axis=0) + fn.sum(axis=0))
    fpr = fp.sum(axis=0) / (fp.sum(axis=0) + tn.sum(axis=0))

    os.makedirs('{}/plots/mixture/'.format(args.TEST.RESULTS.res_dir), exist_ok=True)
    auc, fp = plot_rocs([fpr], [tpr], ["Simultaneous Attacks"], ['red'], '{}/plots/mixture/auroc_{}{}'.format(args.TEST.RESULTS.res_dir, args.DATA_NATURAL.data_name, attack.replace('.', '')), show=False)

    return auc, fp
