import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    clf = GDA()
    clf.fit(x_train, y_train)

    y_hat = clf.predict(x_eval)
    np.savetxt(pred_path, y_hat > 0.5, fmt='%.1f')
    util.plot(x_train, y_train, clf.theta, pred_path.split('.')[0] + '.png')

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape[0], x.shape[1]
        self.theta = np.zeros(n+1)
        phi = (y == 1).mean()
        mu0 = x[y == 0, :].sum(axis=0) / (y == 0).sum()
        mu1 = x[y == 1, :].sum(axis=0) / (y == 1).sum()
        x_ = np.array(x)
        x_[y == 0, :] -= mu0
        x_[y == 1, :] -= mu1
        delta = x_.T @ x_ / m 
        inv_delta = np.linalg.inv(delta)
        self.theta[1:] = inv_delta @ (mu1 - mu0)
        self.theta[0] = 0.5 * (mu0 + mu1).dot(inv_delta).dot(mu0 - mu1) - np.log((1 - phi) / phi)
        m, n = x.shape
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta))) 
        # *** END CODE HERE
