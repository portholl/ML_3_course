from typing import List
from copy import deepcopy


def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    return [[X[i][j] for j in range(120, 500, 5)] for i in range(0, len(X), 4)]
    """
    X - двумерный массив вещественных чисел размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n 
    и c 120 по 500 c шагом 5 по оси размерности m
    """
    pass


def sum_non_neg_diag(X: List[List[int]]) -> int:
    count = 0
    sum_minus = 0
    for i in range(len(X[0])):
        if X[i][i] >= 0: count+= X[i][i]
        else: sum_minus+=1
    if sum_minus == len(X[0]): return -1
    return count

    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    pass


def replace_values(X: List[List[float]]) -> List[List[float]]:
    result = [j[:] for j in X]
    for i in range(len(X[0])):
        M = sum([j[i] for j in X]) / len(X)
        for j in range(len(X)):             
            if result[j][i] < 0.25 * M or result[j][i] > 1.5 * M:
                result[j][i] = -1
    return result
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    pass
