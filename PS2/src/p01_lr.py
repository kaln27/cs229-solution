# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y)) + 1e-5 * theta 

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    import matplotlib.pyplot as plt
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    plt.figure()
    plt.plot(Xa[Ya == 1, -2], Xa[Ya == 1, -1], 'bx', linewidth=2, label='positive')
    plt.plot(Xa[Ya == -1, -2], Xa[Ya == -1, -1], 'go', linewidth=2, label='negative')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Dataset A')
    plt.legend()
    plt.savefig('output/p01_ds1_a.png')

    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    plt.figure()
    plt.plot(Xb[Yb == 1, -2], Xb[Yb == 1, -1], 'bx', linewidth=2, label='positive')
    plt.plot(Xb[Yb == -1, -2], Xb[Yb == -1, -1], 'go', linewidth=2, label='negative')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Dataset B')
    plt.legend()
    plt.savefig('output/p01_ds1_b.png')

    logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()
