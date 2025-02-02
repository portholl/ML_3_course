import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    x_counts = np.bincount(x)
    y_counts = np.bincount(y)

    if np.array_equal(x_counts, y_counts):
        return True
    else:
        return False
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    pass


def max_prod_mod_3(x: np.ndarray) -> int:
    date = x[:-1] * x[1:]
    divisible_by_3 = date % 3 == 0
    if np.any(divisible_by_3):
        return np.max(date[divisible_by_3])
    else:
        return -1
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    pass


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.tensordot(image, weights, axes=[2, 0])
    return result
    """
    Сложить каналы изображения с указанными весами.
    """
    pass


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    x_decoded = np.repeat(x[:,0], x[:,1])
    y_decoded =  np.repeat(y[:,0], y[:,1])
    if len(x_decoded) != len(y_decoded):
        return -1
    result = np.dot(x_decoded, y_decoded)
    return result
    
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    pass


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_norm = np.linalg.norm(X, axis=1)[:, None]
    Y_norm = np.linalg.norm(Y, axis=1)[None, :]
    d = np.dot(X, Y.T)
    cosine_distance = d / (X_norm * Y_norm)
    cosine_distance[(X_norm == 0) | (Y_norm == 0)] = 1

    return cosine_distance
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    pass