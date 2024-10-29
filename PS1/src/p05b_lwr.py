import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    pred_valid = model.predict(x_valid)
    mse = ((y_valid - pred_valid) ** 2).mean()
    print(f'MSE on valid set: {mse}')
    # Plot validation predictions on top of training set
    plt.figure()
    # No need to save predictions
    # Plot data
    plt.scatter(x_train[:, 1], y_train, c = 'b', marker='x', label='train')
    plt.scatter(x_valid[:, 1], pred_valid, c = 'r', marker='o', label='pred_valid')
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('output/p05b.png')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        pred = np.zeros(m)
        # TODO: Why is that for every x_i the W matrix is different, and you need to re-calculate the theta parameters
        for i in range(m):
            W = np.diag(1/2 * np.exp(- ((x[i] - self.x) ** 2).sum(axis=1) / (2 * self.tau ** 2)))
            pred[i] = np.linalg.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y @ x[i]

        return pred 
        # *** END CODE HERE ***
