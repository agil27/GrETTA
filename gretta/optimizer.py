import numpy as np
import torch
from gretta.augmentations import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class VanillaES(object):
    ''' A vanilla(Monte-Carlo) gradient estimator'''

    def __init__(self, sigma=0.1, beta=1.0, n=4, num_samples=20):
        '''
        :param sigma, beta: hyper-params for random sampling
        :param num_samples: number of random sampled steps
        :param n: size of the variable to be optimized
        '''
        super(VanillaES, self).__init__()
        self.sigma = sigma
        self.num_samples = num_samples
        self.beta = beta
        self.scale = self.sigma / np.sqrt(n)

    def vanilla_sample(self, batch_size, data_size):
        '''
        a simple monte-carlo sampler for the random step direction
        :param data_size: length of the vector to be optimized
        :param scale: the size of the hypothesized Gaussian distribution
        '''
        dist = torch.distributions.Normal(torch.zeros(data_size), torch.ones(data_size))
        sample = dist.sample((batch_size,))
        return sample * self.scale

    def step(self, vars, target):
        '''
        :param epoch: number of current iteration
        :param vars: variables to be optimized
        :param target: the target function w.r.t vars to be minimized
        :param verbal: whether to output debug strings
        :return: estimated grad
        '''
        batch_size, vars_size = vars.shape
        vars_pos_concat = []
        vars_neg_concat = []
        epsilon_concat = []
        for _ in range(self.num_samples):
            epsilon = self.vanilla_sample(batch_size, vars_size)
            epsilon = epsilon.to(device)
            epsilon_concat.append(epsilon)
            vars_pos = (vars + epsilon).clamp(0, 1)
            vars_neg = (vars - epsilon).clamp(0, 1)
            vars_pos_concat.append(vars_pos)
            vars_neg_concat.append(vars_neg)
        vars_pos_concat = torch.cat(vars_pos_concat, dim=0)
        vars_neg_concat = torch.cat(vars_neg_concat, dim=0)
        epsilon_concat = torch.cat(epsilon_concat, dim=0)
        target_pos = target(vars_pos_concat)
        target_neg = target(vars_neg_concat)
        grad = self.beta / (2 * (self.sigma ** 2)) * (target_pos - target_neg) * epsilon_concat
        grad = grad.reshape((self.num_samples, batch_size, vars_size))
        grad = grad.mean(dim=0)
        return grad


class SurrogateES(object):
    def __init__(self, sigma=0.1, alpha=0.5, beta=1.0, n=4, k=1, num_samples=20):
        super(SurrogateES, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.a = sigma * np.sqrt(alpha / float(n))
        self.c = sigma * np.sqrt((1.0 - alpha) / float(k))
        self.k = k
        self.num_samples = num_samples

    def ges_sample(self, subspace):
        '''A advanced sampler based on QR decomposition'''
        batch_size, data_size = subspace.shape
        dist_full = torch.distributions.Normal(torch.zeros(data_size), torch.ones(data_size))
        epsilon_full = dist_full.sample((batch_size,)).to(device)
        dist_subspace = torch.distributions.Normal(torch.zeros(self.k), torch.ones(self.k))
        epsilon_subspace = dist_subspace.sample((batch_size,)).to(device)
        epsilon = self.a * epsilon_full
        norm = torch.sqrt((subspace ** 2).sum(dim=1)).unsqueeze(0).T
        q = subspace / norm
        epsilon = epsilon + self.c * epsilon_subspace * q
        return epsilon

    def step(self, vars, target, surrogate):
        '''
        :param surrogate: function to compute surrogate gradients (maybe biased but relevant)
        '''
        surr_grad = surrogate(vars)
        batch_size, vars_size = vars.shape
        vars_pos_concat = []
        vars_neg_concat = []
        epsilon_concat = []
        for _ in range(self.num_samples):
            epsilon = self.ges_sample(surr_grad)
            epsilon = epsilon.to(device)
            epsilon_concat.append(epsilon)
            vars_pos = (vars + epsilon).clamp(0, 1)
            vars_neg = (vars - epsilon).clamp(0, 1)
            vars_pos_concat.append(vars_pos)
            vars_neg_concat.append(vars_neg)
        vars_pos_concat = torch.cat(vars_pos_concat, dim=0)
        vars_neg_concat = torch.cat(vars_neg_concat, dim=0)
        epsilon_concat = torch.cat(epsilon_concat, dim=0)
        target_pos = target(vars_pos_concat)
        target_neg = target(vars_neg_concat)
        grad = self.beta / (2 * (self.sigma ** 2)) * (target_pos - target_neg) * epsilon_concat
        grad = grad.reshape((self.num_samples, batch_size, vars_size))
        grad = grad.mean(dim=0)
        return grad


class TTATarget(object):
    def __init__(self, model, inputs, labels, criterion,
                 normalize=None, output_indices=None, label_indices=None):
        super(TTATarget, self).__init__()
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.criterion = criterion
        self.normalize = normalize
        self.output_indices = output_indices
        if label_indices is not None:
            self.labels = self.labels[:, label_indices]

    def __call__(self, levels):
        x = apply_transform(self.inputs, levels)
        if self.normalize is not None:
            x = self.normalize(x)
        with torch.no_grad():
            f = self.model(x)
            if self.output_indices is not None:
                f = f[:, self.output_indices]
            l = self.criterion(f, self.labels)
        return l


class StudentSurrogate(object):
    def __init__(self, student, inputs, labels, criterion,
                 normalize=None, output_indices=None, label_indices=None):
        super(StudentSurrogate, self).__init__()
        self.student = student
        self.inputs = inputs
        self.labels = labels
        self.criterion = criterion
        self.normalize = normalize
        self.output_indices = output_indices
        if label_indices is not None:
            self.labels = self.labels[:, label_indices]

    def __call__(self, levels):
        levels_copy = levels.detach()
        levels_copy.requires_grad_(True)
        x = apply_transform(self.inputs, levels_copy)
        if self.normalize is not None:
            x = self.normalize(x)
        bias_f = self.student(x)
        if self.output_indices is not None:
            bias_f = bias_f[:, self.output_indices]
        error = self.criterion(bias_f, self.labels)
        error.backward()
        surr_grad = levels_copy.grad.data
        return surr_grad
