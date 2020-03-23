import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

#
# core code from: https://github.com/hs2k/pytorch-smoothgrad
# thanks!
#

class VanillaGrad(object):
    
    def __init__(self, pretrained_model, device=0):
        self.pretrained_model = pretrained_model
        self.device = device

    def __call__(self, x, label=None):
        x.requires_grad_()
        x.retain_grad() 
        output = self.pretrained_model(x)
        if label is None:
            label = torch.argmax(output)
        one_hot = torch.zeros((1, output.size()[-1]), \
                  dtype=torch.float32, device=self.device)
        one_hot[0, label] = 1
        one_hot = torch.sum(one_hot * output)
        one_hot.backward()
        grad = x.grad
        return grad

class IntegratedGradients(object):

    def __init__(self, pretrained_model, steps=100, device=0):
        self.pretrained_model = pretrained_model
        self.steps = steps
        self.device = device

    def __call__(self, x, baseline=None, label=None):
        output = self.pretrained_model(x)
        if label is None:
            label = torch.argmax(output)
        baseline = torch.zeros_like(x).to(self.device)
        scaled_inputs = [baseline + (float(i) / self.steps) * (x - baseline) \
                                    for i in range(0, self.steps + 1)]
        scaled_inputs = torch.cat(scaled_inputs, dim=1)
        scaled_inputs.requires_grad_()
        scaled_inputs.retain_grad()
        
        output = self.pretrained_model(scaled_inputs)
        one_hot = torch.zeros(output.size(), dtype=torch.float32, device=self.device)
        one_hot[:, label] = 1
        one_hot = torch.sum(one_hot * output, dim=1).mean()
        one_hot.backward()
        grad = scaled_inputs.grad
        avg_grad = torch.mean(grad, dim=1, keepdim=True)
        grad = (x - baseline) * avg_grad
        return grad

class SmoothGrad(object):

    def __init__(self, pretrained_model, device=0, noise_level=0.15, n_samples=50, magnitude=True):
        self.pretrained_model = pretrained_model
        self.device = device
        self.noise_level = noise_level
        self.n_samples = n_samples
        self.magnitude = magnitude

    def __call__(self, x, label=None):
        output = self.pretrained_model(x)
        if label is None:
            label = torch.argmax(output)
        stdev = self.noise_level * (torch.max(x) - torch.min(x))
        mu =  torch.zeros_like(x)
        scaled_inputs = [x + torch.normal(mu, stdev) for i in range(self.n_samples)] 
        scaled_inputs = torch.cat(scaled_inputs, dim=1)
        scaled_inputs.requires_grad_()
        scaled_inputs.retain_grad()
        
        output = self.pretrained_model(scaled_inputs)
        one_hot = torch.zeros(output.size(), dtype=torch.float32, device=self.device)
        one_hot[:, label] = 1
        one_hot = torch.sum(one_hot * output, dim=1).mean()
        one_hot.backward()
        grad = scaled_inputs.grad
        if self.magnitude:
            grad = grad ** 2
        avg_grad = torch.mean(grad, dim=1, keepdim=True)
        return avg_grad

