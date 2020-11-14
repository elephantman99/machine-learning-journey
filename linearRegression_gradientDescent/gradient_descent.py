# -*- coding: utf-8 -*-
"""Gradient Descent"""
from linearRegression_gradientDescent import costs as cts
import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # compute gradient and loss
    e = y - np.dot(tx, w)
    N = len(y)
    gradient = - (1 / N) * np.dot(np.transpose(tx), e)
    return gradient


def gradient_descent(y, tx, initial_w, max_iters, gamma, threshold=10**(-5)):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [cts.compute_mse(y, tx, initial_w)]
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        grad = compute_gradient(y, tx, w)
        # update rule
        w = w - gamma*grad
        # compute loss
        loss = cts.compute_mse(y, tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
#         print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#               bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws