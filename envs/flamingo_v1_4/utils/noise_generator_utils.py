import numpy as np
from scipy.stats import truncnorm


def constant_noisy_data(data, noise):
    return data + noise

def uniform_noisy_data(data, lower, upper):
    noise = np.random.uniform(lower, upper, size=data.shape)
    return data + noise

def gaussian_noisy_data(data, mean, std):
    noise = np.random.normal(mean, std, size=data.shape)
    return data + noise

def truncated_gaussian_noisy_data(data, mean, std, lower, upper):
    a = (lower - mean) / std
    b = (upper - mean) / std
    noise = truncnorm.rvs(a, b, loc=mean, scale=std, size=data.shape)
    return data + noise
