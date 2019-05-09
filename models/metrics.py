import numpy as np

def smape(Ft: np.ndarray, At: np.ndarray):
    """SMAPE"""
    return np.mean(np.sum(np.abs(Ft - At) / ((Ft + At) / 2)))
