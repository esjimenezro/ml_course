import numpy as np


def feature_normalization(X):
    """
    Mean/standard deviation normalization.
    
    :param np.ndarray X: Feature matrix to be normalized.
    :return: np.ndarray of normalized feature matrix, np.ndarray of means and np.ndarray of standard deviation.
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    return (X - mu) / sigma, mu, sigma


def grad_desc(grad, x0, args, alpha=0.1, save_steps=True, grad_tol=1e-6, max_iter=1e6):
    """
    Gradient descent algorithm.
    
    :param function grad: Callable of the gradient of the function to be minimized.
    :param np.ndarray x0: Initial guess.
    :param float alpha: Learning rate.
    :param bool save_steps: Whether to save the gradient descent steps or not.
    :param float grad_tol: Early stoping tolerance criterion for the gradient.
    :param int max_iter: Maximum number of iterations.
    :return: np.ndarray of gradient descent steps, or the final convergence point.
    """
    i = 0
    x = x0
    if save_steps:
        steps = [x]
        while (np.linalg.norm([grad(steps[-1], *args)], 2) > grad_tol and i < max_iter):
            i += 1
            steps.append(steps[-1] - alpha * grad(steps[-1], *args))
        return np.array(steps)
    else:
        while (np.linalg.norm([grad(x, *args)], 2) > grad_tol and i < max_iter):
            i += 1
            x -= alpha * grad(x, *args)
        return x
    

if __name__ == '__main__':
    pass