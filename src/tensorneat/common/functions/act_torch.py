import torch

SCALE = 3


def scaled_sigmoid_(z):
    return torch.sigmoid(z) * SCALE


def sigmoid_(z):
    return torch.sigmoid(z)


def scaled_tanh_(z):
    return torch.tanh(z) * SCALE


def tanh_(z):
    return torch.tanh(z)


def sin_(z):
    return torch.sin(z)


def relu_(z):
    return torch.relu(z)


def lelu_(z):
    return torch.where(z > 0, z, 0.005 * z)


def identity_(z):
    return z


def inv_(z):
    safe = torch.where(z > 0, torch.clamp(z, min=1e-7), torch.clamp(z, max=-1e-7))
    return 1.0 / safe


def log_(z):
    return torch.log(torch.clamp(z, min=1e-7))


def exp_(z):
    return torch.exp(z)


def abs_(z):
    return torch.abs(z)