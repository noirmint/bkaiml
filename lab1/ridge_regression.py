import numpy as np
import pandas as pd


def get_data():
    df = pd.read_csv('data/x28.txt', skiprows=72,
                     sep=r'\s+', header=None, index_col=0)
    df.columns = [f'A{i}' for i in range(1, 16)] + ['B']
    df = df.reset_index(drop=True)
    return df.iloc[:, :-1], df.iloc[:, -1]


def normalize_and_add_ones(X):
    X = np.array(X)
    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    X_normalized = (X - X_min) / (X_max - X_min)
    ones = np.ones(X_normalized.shape[0])
    return np.column_stack((ones, X_normalized))


class RidgeRegression:
    def __init__(self):
        pass

    def fit(self, X, y, Lambda):
        assert len(X.shape) == 2 and X.shape[0] == y.shape[0]
        W = np.linalg.inv(X.T @ X + Lambda * np.eye(X.shape[1])) @ X.T @ y
        return W

    def fit_gradient_descent(self, X_train, y_train, Lambda, learning_rate, max_num_epochs=100, batch_size=128):
        W = np.random.randn(X_train.shape[1])
        last_loss = 1e9
        for epoch in range(max_num_epochs):
            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train = X_train[arr]
            y_train = y_train[arr]
            total_minibatches = int(np.ceil(X_train.shape[0] / batch_size))

            for i in range(total_minibatches):
                index = i * batch_size
                X_batch = X_train[index:index + batch_size]
                y_batch = y_train[index:index + batch_size]
                gradient = X_batch.T @ (X_batch @ W - y_batch) + Lambda * W
                W -= learning_rate * gradient

            new_loss = self.compute_RSS(self.predict(W, X_train), y_train)

            if np.abs(new_loss - last_loss) <= 1e-5:
                break
            last_loss = new_loss

        return W

    def predict(self, W, X_new):
        X_new = np.array(X_new)
        y_new = X_new @ W
        return y_new

    def compute_RSS(self, y_new, y_pred):
        loss = 1. / y_new.shape[0] * np.sum((y_new - y_pred) ** 2)
        return loss

    def get_the_best_Lambda(self, X_train, y_train):
        def cross_validation(num_folds, Lambda):
            row_ids = np.array(range(X_train.shape[0]))
            valid_ids = np.split(
                row_ids[: len(row_ids) - len(row_ids) % num_folds], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1],
                                      row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]]
                         for i in range(num_folds)]
            average_RSS = 0
            for i in range(num_folds):
                valid_part = {
                    'X': X_train[valid_ids[i]], 'y': y_train[valid_ids[i]]}
                train_part = {
                    'X': X_train[train_ids[i]], 'y': y_train[train_ids[i]]}
                W = self.fit(train_part['X'], train_part['y'], Lambda)
                y_pred = self.predict(W, valid_part['X'])
                average_RSS += self.compute_RSS(valid_part['y'], y_pred)

            return average_RSS / num_folds

        def range_scan(best_Lambda, best_RSS, Lambdas):
            for current_Lambda in Lambdas:
                current_RSS = cross_validation(5, current_Lambda)
                if current_RSS < best_RSS:
                    best_RSS = current_RSS
                    best_Lambda = current_Lambda
            return best_Lambda, best_RSS

        best_Lambda, best_RSS = range_scan(
            best_Lambda=0, best_RSS=100000 ** 2, Lambdas=range(50))
        Lambdas = [
            k * 1. / 1000 for k in range(max(0, (best_Lambda - 1) * 1000), (best_Lambda + 1) * 1000)]
        best_Lambda, best_RSS = range_scan(best_Lambda, best_RSS, Lambdas)

        return best_Lambda


if __name__ == '__main__':
    X, y = get_data()
    X = normalize_and_add_ones(X)
    y = (y - y.min()) / (y.max() - y.min())
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]
    model = RidgeRegression()
    Lambda = model.get_the_best_Lambda(X_train, y_train)
    print("Best Lambda:", Lambda)
    W = model.fit(X_train, y_train, Lambda)
    print("Weights:", W)
    RSS = model.compute_RSS(y_test, model.predict(W, X_test))
    print("RSS:", RSS)
