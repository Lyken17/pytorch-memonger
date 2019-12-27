from math import sqrt, log
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from .checkpoint import checkpoint


def reforwad_momentum_fix(origin_momentum):
    return (1 - sqrt(1 - origin_momentum))


class SublinearSequential(nn.Sequential):
    def __init__(self, *args):
        super(SublinearSequential, self).__init__(*args)
        self.reforward = False
        self.momentum_dict = {}
        self.set_reforward(True)

    def set_reforward(self, enabled=True):
        if not self.reforward and enabled:
            print("Rescale BN Momemtum for re-forwarding purpose")
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    self.momentum_dict[n] = m.momentum
                    m.momentum = reforwad_momentum_fix(self.momentum_dict[n])
        if self.reforward and not enabled:
            print("Re-store BN Momemtum")
            for n, m in self.named_modules():
                if isinstance(m, _BatchNorm):
                    m.momentum = self.momentum_dict[n]
        self.reforward = enabled

    def forward(self, input):
        if self.reforward:
            return self.sublinear_forward(input)
        else:
            return self.normal_forward(input)

    def normal_forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def sublinear_forward(self, input):
        def run_function(start, end, functions):
            def forward(*inputs):
                input = inputs[0]
                for j in range(start, end + 1):
                    input = functions[j](input)
                return input

            return forward

        functions = list(self.children())
        segments = int(sqrt(len(functions)))
        segment_size = len(functions) // segments
        # the last chunk has to be non-volatile
        end = -1
        if not isinstance(input, tuple):
            inputs = (input,)
        for start in range(0, segment_size * (segments - 1), segment_size):
            end = start + segment_size - 1
            inputs = checkpoint(run_function(start, end, functions), *inputs)
            if not isinstance(inputs, tuple):
                inputs = (inputs,)
        # output = run_function(end + 1, len(functions) - 1, functions)(*inputs)
        output = checkpoint(run_function(end + 1, len(functions) - 1, functions), *inputs)
        return output
