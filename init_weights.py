import torch
import torch.nn as nn
from functools import partial


def init_model_weights(model, init_type:str):
        init_types = {"uniform":torch.nn.init.uniform,
                      "xavier":torch.nn.init.xavier_uniform,
                      "normal":torch.nn.init.normal_,
                      "ones":torch.nn.init.ones_,
                      "zeros":torch.nn.init.zeros_,
                      "eye":torch.nn.init.eye,
                      "xavier_uniform":torch.nn.init.xavier_uniform_,
                      "kaiming_uniform":torch.nn.init.kaiming_uniform_,
                      "kaiming_normal":torch.nn.init.kaiming_normal_,
                      "orthogonal":torch.nn.init.orthogonal_}
        init_type = init_types[init_type]
        init_func = partial(init_layer_weights, func=init_type)
        model.apply(init_func)

def init_layer_weights(layer, func):
    try:
        func(layer.weight)
    except AttributeError:
        return