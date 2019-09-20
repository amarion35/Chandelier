import torch

def binary_accuracy(y_true, y_pred):
    return torch.eq(torch.ge(y_pred, 0.5), y_pred.type(torch.bool)).type(torch.float).mean()

def categorical_accuracy(y_true, y_pred):
    return torch.eq(y_true.argmax(dim=-1), y_pred.argmax(dim=-1)).type(torch.float).mean()

def sparse_categorical_accuracy(y_true, y_pred):
    return torch.eq(y_true, y_pred.argmax(dim=-1)).type(torch.float).mean()