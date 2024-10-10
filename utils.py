import mlx.core as mx
import mlx.nn as nn
import numpy as np


EPSILON = 1e-2

def linear_quantize(samples, q_levels):
    samples = mx.array(samples)  # Ensure samples is an MLX array
    samples -= mx.min(samples, axis=-1, keepdims=True)
    samples /= mx.max(samples, axis=-1, keepdims=True)
    samples *= q_levels - EPSILON
    samples += EPSILON / 2
    return mx.astype(samples, mx.int32)

def linear_dequantize(samples, q_levels):
    return mx.astype(samples, mx.float32) / (q_levels / 2) - 1

def q_zero(q_levels):
    return q_levels // 2