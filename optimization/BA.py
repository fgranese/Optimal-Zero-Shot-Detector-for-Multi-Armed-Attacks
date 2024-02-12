import torch
class blahutArimoto:
    def __init__(self, channel: torch.Tensor, max_iter: int = int(1e3), log_base: float = 2, threshold: float = 1e-12, device=torch.device('cpu')):
        self.channel = channel
        self.max_iter = max_iter
        self.log_base = log_base
        self.threshold = threshold
        self.device = device

    def compute(self, verbose: bool = False):
        # C = number of classification labels, K = number of detectors, N = number of samples
        # channel = N x K x C
        # weights = N X K x 1
        num_samples = self.channel.shape[0]
        num_detectors = self.channel.shape[1]
        num_classes = self.channel.shape[2]

        if verbose:
            print(f"Channel:\n {self.channel}")
            print(f"\nChannel shape: {self.channel.shape}")

        # assert that sum of self.channels along dim=2 is 1 for all samples and detectors
        assert torch.allclose(torch.sum(self.channel, dim=2),
                              torch.ones(num_samples, num_detectors)), "Channel probabilities do not sum to 1"

        # create a tensor for weights of shape N x K x 1 where each element is 1/K
        weights = torch.ones(num_samples, num_detectors, 1) / num_detectors

        weights = weights.to(self.device)
        self.channel = self.channel.to(self.device)

        if verbose:
            print(f"\nWeight shape {weights.shape}")

        for iter_id in range(self.max_iter):
            # compute q as the product of weights and channel for each of the N samples
            if verbose:
                print(f"\nIteration {iter_id}")
            q = torch.mul(weights, self.channel)
            q = q / torch.sum(q, dim=1, keepdim=True)
            if verbose:
                print(f"\nq shape: {q.shape}")

            w1 = torch.prod(torch.pow(q, self.channel), dim=2, keepdim=True)
            w1 = w1 / torch.sum(w1, dim=1, keepdim=True)
            if verbose:
                print(f"\nw1 shape: {w1.shape}")

            tolerance = torch.linalg.norm(w1 - weights)
            weights = w1
            if tolerance < self.threshold:
                break

        # print(tolerance, self.threshold)
        # print(f"\nFinal iteration: {iter_id}")
        return weights.squeeze(dim=-1).detach().cpu()

