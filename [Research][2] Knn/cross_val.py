import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_size = num_objects // num_folds
    remainder = num_objects % num_folds
    indices = np.arange(num_objects)

    folds = []
    start = 0
    for i in range(num_folds):
        end = start + fold_size + (remainder if i == num_folds - 1 else 0)
        fold = indices[start:end]
        folds.append(fold)
        start = end

    split_pairs = []
    for i in range(num_folds):
        train_indices = np.concatenate(
            [folds[j] for j in range(num_folds) if j != i])
        val_indices = folds[i]
        split_pairs.append((train_indices, val_indices))

    return split_pairs
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    pass


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    scores = {}
    n_neighbors_params = parameters.get('n_neighbors')
    metrics_params = parameters.get('metrics')
    weights_params = parameters.get('weights')
    normalizers = parameters.get('normalizers')
    for normalizer, normalizer_name in normalizers:
        for n_neighbors in n_neighbors_params:
            for metric in metrics_params:
                for weight in weights_params:
                    knn = knn_class(n_neighbors=n_neighbors,
                                    metric=metric, weights=weight)
                    fold_scores = []

                    for train_idx, val_idx in folds:
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        if normalizer:
                            normalizer.fit(X_train)
                            X_train = normalizer.transform(X_train)
                            X_val = normalizer.transform(X_val)

                        knn.fit(X_train, y_train)
                        y_pred = knn.predict(X_val)

                        fold_scores.append(score_function(y_val, y_pred))

                    scores[(normalizer_name, n_neighbors, metric,
                            weight)] = np.mean(fold_scores)

    return scores
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    pass
