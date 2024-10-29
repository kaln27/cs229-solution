import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path, 'y', True)
    _, t_train = util.load_dataset(train_path, 't')

    x_valid, y_valid = util.load_dataset(valid_path, 'y', True)
    _, t_valid = util.load_dataset(valid_path, 't')

    x_test, y_test = util.load_dataset(test_path, 'y', True)
    _, t_test = util.load_dataset(test_path, 't')
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    model_c = LogisticRegression()
    model_c.fit(x_train, t_train)
    pred_c = model_c.predict(x_test)
    np.savetxt(pred_path_c, pred_c > 0.5, fmt='%.1f')
    util.plot(x_test, t_test, model_c.theta, pred_path_c.split('.')[0] + '.png')

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    model_d = LogisticRegression()
    model_d.fit(x_train, y_train)
    pred_d = model_d.predict(x_test)
    np.savetxt(pred_path_d, pred_d > 0.5, fmt='%.1f')
    util.plot(x_test, t_test, model_d.theta, pred_path_d.split('.')[0] + '.png')

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    alpha = model_d.predict(x_valid[y_valid==1]).mean()
    np.savetxt(pred_path_e, pred_d / alpha > 0.5, fmt='%.1f')
    util.plot(x_test, t_test, model_d.theta, pred_path_e.split('.')[0] + '.png', base = np.log(alpha / (2 - alpha)))

    # *** END CODER HERE
