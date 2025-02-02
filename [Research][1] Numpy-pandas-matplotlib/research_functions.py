from collections import Counter
from typing import List


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    def count_elements(vector):
        element_counts = {}
        for element in vector:
            if element in element_counts:
                element_counts[element] += 1
            else:
                element_counts[element] = 1
        return element_counts

    x_counts = count_elements(x)
    y_counts = count_elements(y)
    if x_counts == y_counts:
        return True
    else:
        return False
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    pass


def max_prod_mod_3(x: List[int]) -> int:
    m = -1
    for i in range(len(x) - 1):
            if x[i] % 3 == 0 or x[i+1] % 3 == 0:
                if x[i] * x[i+1] > m: m = x[i] * x[i+1]
    return m
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    pass


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    result = [[0 for _ in range(len(image[0]))] for _ in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[i])):
            for k in range(len(image[i][0])):
                image[i][j][k] *= weights[k]
                result[i][j]  += image[i][j][k]
    return result
        
    """
    Сложить каналы изображения с указанными весами.
    """
    pass


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    x_decoded = []
    for i, count in x:
        x_decoded.extend([i] * count)
    y_decoded = []
    for i, count in y:
        y_decoded.extend([i] * count)
    if len(x_decoded) != len(y_decoded):
        return -1
    result = 0
    for i in range(len(x_decoded)):
        result += x_decoded[i] * y_decoded[i]
    
    return result
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """

    pass


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    result = [[0 for _ in range(len(Y))] for _ in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            d = sum(a * b for a, b in zip(X[i], Y[j]))
            X_norm = sum(a ** 2 for a in X[i]) ** 0.5
            Y_norm = sum(b ** 2 for b in Y[j]) ** 0.5

            if X_norm == 0 or Y_norm == 0:
                result[i][j] = 1
            else:
                result[i][j] = d / (X_norm * Y_norm) 

    return result

    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    pass