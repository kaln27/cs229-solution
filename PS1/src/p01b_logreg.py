import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    y_hat = clf.predict(x_eval)
    np.savetxt(pred_path, y_hat > 0.5, fmt='%.1f')
    util.plot(x_train, y_train, clf.theta, pred_path.split('.')[0] + '.png')
    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    @staticmethod
    def sigmod(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        m, n= x.shape[0], x.shape[1]
        if self.theta == None:
            self.theta = np.zeros(n)
        diff_ = np.full(n, np.inf)
        while np.linalg.norm(diff_, 1) >= self.eps:
            hx = self.sigmod(x.dot(self.theta))
            grad = x.T.dot(hx - y) / m
            H = (x.T * hx * (1 - hx)) @ x / m
            diff_ = np.linalg.inv(H) @ grad
            self.theta -= diff_ 
            
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        return self.sigmod(x.dot(self.theta)) 
        # *** END CODE HERE ***
