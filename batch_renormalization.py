import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class BatchRenorm1d(nn.Module):
    def __init__(self, num_features, r_d_func, eps=1e-5, momentum=0.1, affine=True):
        super(BatchRenorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.r_d_func = r_d_func
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.running_mean.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def forward(self, input, itr):
        self._check_input_dim(input)
        mean = input.mean(0).expand_as(input)
        var = input.var(0).expand_as(input) + self.eps
        bn = (input-mean) / var
        r_max, d_max = self.r_d_func(itr)
        r = (var/Variable(self.running_var.unsqueeze(0).expand_as(var))).clamp(1/r_max, r_max)
        d = ((mean-Variable(self.running_mean.unsqueeze(0).expand_as(mean))) / \
                Variable(self.running_var.unsqueeze(0).expand_as(var))).clamp(-d_max, d_max)
        self.running_mean = self.running_mean + self.momentum * (mean.data.mean(0)-self.running_mean)
        self.running_var = self.running_var + self.momentum * (var.data.mean(0)-self.running_var)
        r = Variable(r.data)
        d = Variable(d.data)
        return bn * r + d

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))
