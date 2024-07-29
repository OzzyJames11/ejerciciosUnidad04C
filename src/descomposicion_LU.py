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

# ####################################################################
def descomposicion_LU(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Realiza la descomposición LU de una matriz cuadrada A.
    [IMPORTANTE] No se realiza pivoteo.

    ## Parameters

    ``A``: matriz cuadrada de tamaño n-by-n.

    ## Return

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior. Se obtiene de la matriz ``A`` después de aplicar la eliminación gaussiana.
    """

    A = np.array(
        A, dtype=float
    )  # convertir en float, porque si no, puede convertir como entero

    assert A.shape[0] == A.shape[1], "La matriz A debe ser cuadrada."
    n = A.shape[0]

    L = np.zeros((n, n), dtype=float)

    for i in range(0, n):  # loop por columna

        # --- deterimnar pivote
        if A[i, i] == 0:
            raise ValueError("No existe solución única.")

        # --- Eliminación: loop por fila
        L[i, i] = 1
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]

            L[j, i] = m


    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    return L, A


# ####################################################################
def resolver_LU(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante la descomposición LU.

    ## Parameters

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior.

    ``b``: vector de términos independientes.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """

    # Completar
    print("Calculando y")
    
     # --- Sustitución hacia atrás
    n = L.shape[0]
    solucion = np.zeros(n)
    solucion[0] = b[0] / L[0, 0]

    for i in range(1,n):
        suma = 0
        for j in range(i):
            suma += L[i, j] * solucion[j]
        solucion[i] = (b[i] - suma) / L[i, i]
    
    print("y")
    print(solucion)
    print("Verificación Ly=b:")
    verif_y = np.matmul(L, solucion)
    print(verif_y)
    
    print("Calculando x") 
    # --- Sustitución hacia atrás
    solucion_f = np.zeros(n, dtype=float)
    solucion_f[n-1] = solucion[n-1] / U[n-1, n-1]

    for i in range(n-2,-1,-1):
        suma = 0
        for j in range(i+1, n):
            suma += U[i, j] * solucion_f[j]
        solucion_f[i] = (solucion[i] - suma) / U[i, i]
    
    print("x")
    print(solucion_f)
    print("Verificación Ux=y:")
    verif_x = np.matmul(U, solucion_f)
    
    print(verif_x)
                 
    return