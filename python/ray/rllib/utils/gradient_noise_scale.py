from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def gradient_noise_scale(cur_batch_size, cur_grads, true_grads):
    """Estimator for the current gradient noise scale.

    This implements Equation 2.10 in https://arxiv.org/pdf/1812.06162.pdf
    to estimate B_simple.

    Arguments:
        cur_batch_size (Tensor): Size of the current minibatch.
        cur_grads (list): Gradient tensors for the current minibatch.
        true_grads (list): Estimate of the true gradient (i.e., grad tensors
            averaged over many batches).
    """

    assert len(cur_grads) == len(true_grads), (cur_grads, true_grads)

    G_diff = [g_est - g for (g_est, g) in zip(cur_grads, true_grads)]
    B_simple = cur_batch_size * (
        tf.square(tf.global_norm(G_diff)) /
        tf.square(tf.global_norm(true_grads)))

    return B_simple
