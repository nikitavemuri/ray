from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def gradient_noise_scale(grads):
    # Calculate gradient element variances
    grad_va = []
    for g in grads:
        _, g_va = tf.nn.moments(g, axes=[0], keep_dims=True)
        g_va = tf.squeeze(g_va, axis=0)
        if len(g_va.shape) == 0:
            g_va = tf.expand_dims(g_va, 0)
        grad_va.append(g_va)
    grad_va = tf.concat(grad_va, axis=0)

    # Equation 2.9 in https://arxiv.org/pdf/1812.06162.pdf
    # B_simple = tr(covar(grad)) / |G|^2 = sum(grad_va) / global_norm(grads)
    B_simple = tf.reduce_sum(grad_va) / tf.global_norm(grads)
    return B_simple
