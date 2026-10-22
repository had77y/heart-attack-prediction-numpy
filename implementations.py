"""
Implementations of the 6 basic machine learning methods for CS-433 Project 1.

Conventions used throughout this file (matching the project spec):
    - y:  numpy array of shape (N,)      -- labels
    - tx: numpy array of shape (N, D)    -- feature matrix (including bias column if needed)
    - w:  numpy array of shape (D,)      -- weight vector
    - Every function returns (w, loss), where w is the LAST weight vector obtained
      (not necessarily the one with lowest loss), and loss is the value of the
      objective evaluated at that w.
    - MSE loss includes the 1/2 factor:  L(w) = (1 / (2N)) * sum((y - tx w)^2)
    - For ridge_regression and reg_logistic_regression, the returned loss does NOT
      include the regularization (penalty) term.
"""

import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm for linear regression using MSE loss.

    Args:
        y: numpy array of shape (N,) -- labels
        tx: numpy array of shape (N, D) -- feature matrix
        initial_w: numpy array of shape (D,) -- initial weight vector
        max_iters: int -- maximum number of iterations
        gamma: float -- learning rate
    """
    w = initial_w
    N = len(y)
    for _ in range(max_iters):
        # Compute the gradient
        e = y - tx @ w
        gradient = -(1 / N) * tx.T @ e
        # Update weights
        w = w - gamma * gradient

    # Compute the loss at the final weights
    e = y - tx @ w
    loss = 1 / 2 * np.mean(e**2)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Stochastic gradient descent algorithm for linear regression using MSE loss.

    Args:
        y: numpy array of shape (N,) -- labels
        tx: numpy array of shape (N, D) -- feature matrix
        initial_w: numpy array of shape (D,) -- initial weight vector
        max_iters: int -- maximum number of iterations
        gamma: float -- learning rate
    """
    w = initial_w
    N = len(y)

    # Perform stochastic gradient descent on one sample at a time
    for _ in range(max_iters):
        i = np.random.randint(N)
        y_i = y[i]
        tx_i = tx[i]
        e = y_i - tx_i @ w
        loss = 1 / 2 * e**2
        gradient = -tx_i * e

        # Update weights
        w = w - gamma * gradient

    e = y - tx @ w
    loss = 1 / 2 * np.mean(e**2)

    return w, loss


def least_squares(y, tx):
    """
    Least squares solution for linear regression.

    Args:
        y: numpy array of shape (N,) -- labels
        tx: numpy array of shape (N, D) -- feature matrix
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - tx @ w
    loss = 1 / 2 * np.mean(e**2)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression solution for linear regression.

    Args:
        y: numpy array of shape (N,) -- labels
        tx: numpy array of shape (N, D) -- feature matrix
        lambda_: float -- regularization parameter
    """
    N = len(y)

    lambda_prime = 2 * N * lambda_
    A = tx.T @ tx + lambda_prime * np.eye(tx.shape[1])
    b = tx.T @ y
    w = np.linalg.solve(A, b)

    e = y - tx @ w
    loss = 1 / 2 * np.mean(e**2)  # should not include regularization term in loss

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N,) -- labels
        tx: numpy array of shape (N, D) -- feature matrix
        initial_w: numpy array of shape (D,) -- initial weight vector
        max_iters: int -- maximum number of iterations
        gamma: float -- learning rate
    """
    w = initial_w
    for _ in range(max_iters):
        # Compute the predictions
        pred = 1 / (1 + np.exp(-tx @ w))
        # Compute the gradient
        gradient = tx.T @ (pred - y) / len(y)
        # Update weights
        w = w - gamma * gradient

    pred = 1 / (1 + np.exp(-tx @ w))
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N,) -- labels
        tx: numpy array of shape (N, D) -- feature matrix
        lambda_: float -- regularization parameter
        initial_w: numpy array of shape (D,) -- initial weight vector
        max_iters: int -- maximum number of iterations
        gamma: float -- learning rate
    """
    w = initial_w
    for _ in range(max_iters):
        # Compute the predictions
        pred = 1 / (1 + np.exp(-tx @ w))
        # Compute the gradient with regularization
        gradient = tx.T @ (pred - y) / len(y) + 2 * lambda_ * w
        # Update weights
        w = w - gamma * gradient

    pred = 1 / (1 + np.exp(-tx @ w))
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    return w, loss
