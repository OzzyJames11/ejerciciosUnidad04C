# -*- coding: utf-8 -*-
"""
Python 3
27 / 07 / 2024
@author: OzzyLoachamin

"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime

import numpy as np

def multiplicar_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiplica dos matrices A y B si sus dimensiones son compatibles.

    ## Parameters

    ``A``: Primera matriz (m x n).
    ``B``: Segunda matriz (n x p).

    ## Return

    ``C``: Matriz resultado de la multiplicación de A y B (m x p).

    ## Raises

    ``ValueError``: Si las dimensiones de las matrices no son compatibles para la multiplicación.
    """
    # Verificar la compatibilidad de las dimensiones
    if A.shape[1] != B.shape[0]:
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación. "
                         "El número de columnas de A debe ser igual al número de filas de B.")
    
    # Multiplicar las matrices
    C = np.dot(A, B)
    
    return C

# ------------------------------------------------------------------------

def determinante(A: np.ndarray) -> float:
    """Calcula el determinante de una matriz cuadrada A.

    ## Parameters

    ``A``: Matriz cuadrada de dimensiones n x n.

    ## Return

    ``determinante``: Determinante de la matriz A.

    ## Raises

    ``ValueError``: Si la matriz no es cuadrada.
    """
    # Verificar que A sea un array de numpy
    if not isinstance(A, np.ndarray):
        raise TypeError("A debe ser un array de numpy.")
    
    # Verificar que A sea una matriz cuadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")

    n = A.shape[0]
    A = A.astype(float)  # Convertir a float para evitar problemas con la división

    # Inicializar el determinante
    det = 1

    for i in range(n):
        # Encontrar el pivote
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if A[max_row, i] == 0:
            return 0  # El determinante es 0 si un pivote es 0

        # Intercambiar filas si es necesario
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            det *= -1  # Cambiar el signo del determinante al intercambiar filas

        # Multiplicar el determinante por el elemento del pivote
        det *= A[i, i]

        # Eliminar las entradas debajo del pivote
        for j in range(i + 1, n):
            if A[j, i] != 0:
                factor = A[j, i] / A[i, i]
                A[j, i:] -= factor * A[i, i:]

    return det

#----------------------------------------------------------------------------------

def inversa(A: np.ndarray) -> np.ndarray:
    """Calcula la inversa de una matriz cuadrada A utilizando el método de Gauss-Jordan.

    ## Parameters

    ``A``: Matriz cuadrada de dimensiones n x n.

    ## Return

    ``inversa``: Matriz inversa de A.

    ## Raises

    ``ValueError``: Si la matriz no es cuadrada o si la matriz es singular (no tiene inversa).
    """
    # Verificar que A sea un array de numpy
    if not isinstance(A, np.ndarray):
        raise TypeError("A debe ser un array de numpy.")
    
    # Verificar que A sea una matriz cuadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")
    
    n = A.shape[0]
    A = A.astype(float)  # Convertir a float para evitar problemas con la división

    # Crear la matriz identidad de tamaño n
    I = np.eye(n)
    
    # Crear una matriz aumentada [A|I]
    AI = np.hstack((A, I))

    for i in range(n):
        # Encontrar el pivote
        max_row = np.argmax(np.abs(AI[i:, i])) + i
        if AI[max_row, i] == 0:
            raise ValueError("La matriz es singular y no tiene inversa.")
        
        # Intercambiar filas si es necesario
        if max_row != i:
            AI[[i, max_row]] = AI[[max_row, i]]
        
        # Normalizar la fila del pivote
        pivot = AI[i, i]
        AI[i, :] = AI[i, :] / pivot
        
        # Eliminar las entradas en otras filas
        for j in range(n):
            if j != i:
                factor = AI[j, i]
                AI[j, :] -= factor * AI[i, :]
    
    # La matriz inversa es la parte derecha de la matriz aumentada
    A_inv = AI[:, n:]
    
    return A_inv