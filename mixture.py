import os
import torch
import numpy as np
from tqdm import tqdm
from utils.utils_models import load_detectors
from scipy.optimize import minimize, Bounds

def compute_probs_detector(intermediate_layer_nat, intermediate_layer_adv, detector):
    detected_nat = torch.sigmoid(detector(intermediate_layer_nat))
    detected_adv = torch.sigmoid(detector(intermediate_layer_adv))

    return np.concatenate((detected_nat.detach().cpu().numpy(), detected_adv.detach().cpu().numpy()))


def optimize(prob_losses_adv, device='cpu', max_iterations=1000):
    # Optimization Eq. (10)
    mutual_informations = np.zeros(prob_losses_adv.shape)

    num_samples = prob_losses_adv.shape[1]
    batch_size = num_samples

    print("Optimization")
    for batch_id in tqdm(range(0, num_samples, batch_size), ncols=70, ascii=True, colour='blue'):
        from optimization.BA import blahutArimoto
        channel_batch = prob_losses_adv.T[batch_id:batch_id+batch_size]
        channel_batch = np.expand_dims(channel_batch, 2)
        channel_batch = np.concatenate((channel_batch, 1-channel_batch), axis=2)
        channel_batch = torch.Tensor(channel_batch)
        opt = blahutArimoto(
            channel=channel_batch, max_iter=max_iterations,
            log_base=2, threshold=1e-12, device=device,
            ).compute(verbose=False).numpy().squeeze()
        mutual_informations[:, batch_id:batch_id+batch_size] = np.asarray(opt).T

    final_prob_adv = np.sum(mutual_informations * prob_losses_adv, axis=0)
    final_prob = np.expand_dims(final_prob_adv, axis=1)

    return final_prob


def mixture_output(args, attack, classifier, device, intermediate_layer_nat, intermediate_layer_adv, n_samples_adv):
    # ---- Load model detectors ----
    losses = list(set(args.TEST.detectors) & {'CE', 'Rao', 'KL', 'g'})
    detectors = [
        load_detectors(args, model=classifier, device=device, loss=losses[i], epsilon=args.TRAIN.PGDi.epsilon)[0][
            -1].eval()
        for i in range(len(losses))]

    # ---- Compute values wrt the detectors ----
    prob_losses_adv = np.zeros((len(detectors), n_samples_adv * 2))

    for i in range(prob_losses_adv.shape[0]):
        prob_losses_adv[i, :] = np.squeeze(
            compute_probs_detector(intermediate_layer_nat, intermediate_layer_adv, detectors[i])
            )

    sota = list(set(args.TEST.detectors) & {'nss', 'fs'})
    if len(sota) != 0:
        prob_losses_adv_and_sota = np.zeros((prob_losses_adv.shape[0] + len(sota), prob_losses_adv.shape[1]))
        prob_losses_adv_and_sota[:prob_losses_adv.shape[0], :] = prob_losses_adv

        for j in range(len(sota)):
            method = sota[j]
            if method == 'nss':
                method_dir = args.TEST.RESULTS.nss_dir
                path_method_file = f"{method_dir}/{args.DATA_NATURAL.data_name}/evaluation/probs_{args.DATA_NATURAL.data_name}{attack}_all.npy"
                proba = np.load(path_method_file)
                proba = proba[:, 1]
            else:
                method_dir = args.TEST.RESULTS.fs_dir
                path_method_file = f"{method_dir}/{args.DATA_NATURAL.data_name}/evaluation/probs_{args.DATA_NATURAL.data_name}{attack}_all.npy"
                proba = np.load(path_method_file)
                proba = (proba - np.min(proba)) / (np.max(proba) - np.min(proba))

            print(path_method_file)

            prob_losses_adv_and_sota[j, :] = proba
        prob_losses_adv = prob_losses_adv_and_sota

    final_prob = optimize(prob_losses_adv, device=device, optimization_type=args.TEST.OPTIMIZATION.type, max_iterations=args.TEST.OPTIMIZATION.max_itaraions)

    prob_natural = final_prob[: len(final_prob) // 2]
    prob_adversarial = final_prob[len(final_prob) // 2:]

    print('Naturals :', np.mean(prob_natural), np.std(prob_natural))
    print('Adversarial :', np.mean(prob_adversarial), np.std(prob_adversarial))

    # ---- Save the probabilities ----
    path_res = '{}{}/all_{}/'.format(args.TEST.RESULTS.res_dir, args.DATA_NATURAL.data_name, args.TRAIN.PGDi.epsilon)
    print('{}/probs_{}_all.npy'.format(path_res, '{}{}'.format(args.DATA_NATURAL.data_name, attack)))
    os.makedirs(path_res, exist_ok=True)
    np.save('{}/probs_{}_all.npy'.format(path_res, '{}{}'.format(args.DATA_NATURAL.data_name, attack)), final_prob)

    return final_prob
