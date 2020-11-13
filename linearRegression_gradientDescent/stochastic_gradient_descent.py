# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
from helpers import batch_iter

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # number of samples
    N = len(y)
    # error vector
    e = y - np.dot(tx, w)
    # stochastic gradient
    stoch_grad = - np.dot(np.transpose(tx), e) / N
    return stoch_grad


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    losses = []
    ws = [initial_w]
    w  = initial_w
    
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size, shuffle=True):
            # compute gradient
            g = compute_stoch_gradient(batch_y, batch_tx, w)
            # compute loss
            loss = compute_loss_mse(batch_y, batch_tx, w)
            # update rule
            w_prev = w
            w = w - gamma * g
            # update lists
            losses += [loss]
            ws += [w_t1]
            
#             print("Stoch Grad Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}"
#.format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws