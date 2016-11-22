__author__ = 'max'

import logging
import sys


def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

########################################
# Reward & Baseline functions
########################################
import numpy as np

discount_const = 0.99
def reward_to_value(rewards):
    values = np.zeros((len(rewards), rewards[0].shape[0]))
    for i in range(len(values)-1,-1,-1):
        values[i] = rewards[i] + values[i+1] * discount_const if i < len(values)-1 else rewards[i]

    return values

bl_decay_const = 0.99
def apply_baseline(values, baselines, masks):
    lengths = np.zeros_like(masks).astype(np.int32)
    lengths[:,:-1] = np.cumsum(masks[:,:0:-1], axis=1)[:,::-1]

    for i in range(len(values)):
        for v, l in zip(values[i], lengths[:,i]):
            baselines[l] = bl_decay_const * baselines[l] + (1.0 - bl_decay_const) * v

    for i in range(len(values)):
        values[i] -= baselines[lengths[:,i]]