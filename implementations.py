"""
Implementations of the 6 basic machine learning methods for CS-433 Project 1.

Conventions used throughout this file (matching the project spec):
    - y:  numpy array of shape (N,)      -- labels
    - tx: numpy array of shape (N, D)    -- feature matrix (including bias column)
    - w:  numpy array of shape (D,)      -- weight vector
    - Every function returns (w, loss), where w is the LAST weight vector obtained
      (not necessarily the one with lowest loss), and loss is the value of the
      objective evaluated at that w.
    - MSE loss includes the 1/2 factor:  L(w) = (1 / (2N)) * sum((y - tx w)^2)
    - For ridge_regression and reg_logistic_regression, the returned loss does NOT
      include the regularization (penalty) term.
"""

import numpy as np


def _mse_loss(y, tx, w):
    """MSE loss with the 1/2 factor: (1 / 2N) * sum((y - tx w)^2)."""
    e = y - tx @ w
    return 1 / 2 * np.mean(e**2)


def _sigmoid(z):
    """Numerically stable sigmoid, equal to 1 / (1 + exp(-z))."""
    return np.exp(-np.logaddexp(0, -z))


def _logistic_loss(y, tx, w):
    """Mean negative log-likelihood for y in {0, 1}, computed in a stable way."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm for linear regression using MSE loss.

    Args:
        y: numpy array of shape (N,) -- labels
        tx: numpy array of shape (N, D) -- feature matrix
        initial_w: numpy array of shape (D,) -- initial weight vector
        max_iters: int -- number of iterations
        gamma: float -- step size

    Returns:
        (w, loss): last weight vector and its MSE loss
    """
    w = initial_w
    N = len(y)
    for _ in range(max_iters):
        gradient = -tx.T @ (y - tx @ w) / N
        w = w - gamma * gradient

    return w, _mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Stochastic gradient descent for linear regression using MSE loss.
    Each step uses a single randomly drawn data point (batch size 1).

    Args:
        y: numpy array of shape (N,) -- labels
        tx: numpy array of shape (N, D) -- feature matrix
        initial_w: numpy array of shape (D,) -- initial weight vector
        max_iters: int -- number of iterations
        gamma: float -- step size

    Returns:
        (w, loss): last weight vector and its MSE loss
    """
    w = initial_w
    N = len(y)
    for _ in range(max_iters):
        i = np.random.randint(N)
        gradient = -tx[i] * (y[i] - tx[i] @ w)
        w = w - gamma * gradient

    return w, _mse_loss(y, tx, w)


def least_squares(y, tx):
    """
    Least squares regression using the normal equations.

    Args:
        y: numpy array of shape (N,) -- labels
        tx: numpy array of shape (N, D) -- feature matrix

    Returns:
        (w, loss): optimal weight vector and its MSE loss
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, _mse_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using the normal equations.
    Objective: (1 / 2N) * ||y - tx w||^2 + lambda_ * ||w||^2

    Args:
        y: numpy array of shape (N,) -- labels
        tx: numpy array of shape (N, D) -- feature matrix
        lambda_: float -- regularization parameter

    Returns:
        (w, loss): weight vector and its MSE loss (without the penalty term)
    """
    N, D = tx.shape
    w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.eye(D), tx.T @ y)
    return w, _mse_loss(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent (labels y in {0, 1}).

    Args:
        y: numpy array of shape (N,) -- labels in {0, 1}
        tx: numpy array of shape (N, D) -- feature matrix
        initial_w: numpy array of shape (D,) -- initial weight vector
        max_iters: int -- number of iterations
        gamma: float -- step size

    Returns:
        (w, loss): last weight vector and its logistic loss
    """
    w = initial_w
    N = len(y)
    for _ in range(max_iters):
        gradient = tx.T @ (_sigmoid(tx @ w) - y) / N
        w = w - gamma * gradient

    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent (labels y in {0, 1}).
    Penalty term: lambda_ * ||w||^2

    Args:
        y: numpy array of shape (N,) -- labels in {0, 1}
        tx: numpy array of shape (N, D) -- feature matrix
        lambda_: float -- regularization parameter
        initial_w: numpy array of shape (D,) -- initial weight vector
        max_iters: int -- number of iterations
        gamma: float -- step size

    Returns:
        (w, loss): last weight vector and its logistic loss (without the penalty)
    """
    w = initial_w
    N = len(y)
    for _ in range(max_iters):
        gradient = tx.T @ (_sigmoid(tx @ w) - y) / N + 2 * lambda_ * w
        w = w - gamma * gradient

    return w, _logistic_loss(y, tx, w)
