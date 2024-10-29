import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    min_mse = np.inf
    best_tau = None

    model = LocallyWeightedLinearRegression(0)
    model.fit(x_train, y_train)
    for i, tau in enumerate(tau_values):
        model.tau = tau
        pred = model.predict(x_valid)
        mse = ((y_valid - pred) ** 2).mean()
        print(f'tau : {tau} \t mse : {mse}')
        if mse < min_mse:
            min_mse = mse
            best_tau = tau
        plt.figure()
        plt.scatter(x_train[:, 1], y_train, c='b', marker='x', label='train')
        plt.scatter(x_valid[:, 1], pred, c='r', marker='o', label='pred_valid')
        plt.title(f'tau : {tau}')
        plt.xlabel('x1')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(f'output/p05c_{i}.png')
    
    # Fit a LWR model with the best tau value
    print(f'Best tau : {best_tau}')
    model.tau = best_tau
    # Run on the test set to get the MSE value
    pred_test = model.predict(x_test)
    mse_test = ((y_valid - pred) ** 2).mean()
    print(f'On test set:\ntau : {best_tau} \t mse : {mse_test}')
    # Save predictions to pred_path
    np.savetxt(pred_path, pred_test)
    # Plot data
    plt.figure()
    plt.scatter(x_test[:, 1], pred_test, c='r', marker='o', label='pred_test')
    plt.scatter(x_train[:, 1], y_train, c='b', marker='x', label='train')
    plt.title('test set\ntau : {}'.format(best_tau))
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('output/p05c_test.png')

    # *** END CODE HERE ***
