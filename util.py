import numpy as np
import torch

index_dtype = np.uint64

def randbit():
    return np.random.randint(2, size=1, dtype=index_dtype)

def make_shuffling(size):
    retval = np.arange(size, dtype=index_dtype)
    np.random.shuffle(retval)
    return retval

def ymd_hms_compact(t):
    return t.isoformat().replace(':', '').replace('-', '').replace('T', '_').split('.')[0]

# compare
# https://discuss.pytorch.org/t/how-to-build-a-view-layer-in-pytorch-for-sequential-models/53958/12
class ViewLayer(torch.nn.Module):
    def __init__(self, shape):
        super(ViewLayer, self).__init__()
        self.shape = shape

    def forward(self, X):
        return X.view(self.shape)

class ScalarMultiplyLayer(torch.nn.Module):
    def __init__(self, scalar):
        super(ScalarMultiplyLayer, self).__init__()
        self.scalar = scalar

    def forward(self, X):
        return X * self.scalar

class IgnoreHiddenAndCellState(torch.nn.Module):
    def __init__(self):
        super(IgnoreHiddenAndCellState, self).__init__()

    def forward(self, X):
        return X[0]