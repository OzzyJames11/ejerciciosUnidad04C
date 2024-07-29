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
def eliminacion_gaussiana_L(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante el método de eliminación gaussiana.

    ## Parameters

    ``A``: matriz aumentada del sistema de ecuaciones lineales. Debe ser de tamaño n-by-(n+1), donde n es el número de incógnitas.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=np.float32)
    b = np.array(b, dtype=float).reshape(-1, 1)
    Ab = np.hstack((A,b))
    assert A.shape[0] == Ab.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    
    n = Ab.shape[0]
    i = 0
    for i in range(0,n): 
        # --- Eliminación: loop por fila
        for j in range(i + 1, n):
            if Ab[j, i] != 0:
                m = Ab[j, i]
                Ab[j, i:] = Ab[j, i:] - m * Ab[i, i:]
            print(f"\n{Ab}")
    if Ab[n - 1, n - 1] == 0:
        print("\nNo existe solución.")
        return None

    # --- Sustitución hacia atrás
    solucion = np.zeros(n, dtype=np.float32)
    solucion[n - 1] = Ab[n - 1, n] / Ab[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += Ab[i, j] * solucion[j]
        solucion[i] = (Ab[i, n] - suma) / Ab[i, i]

    return solucion

#-------------------------------------------------------------------------------------------------

def eliminacion_gaussiana_U(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante el método de eliminación gaussiana.

    ## Parameters

    ``A``: matriz aumentada del sistema de ecuaciones lineales. Debe ser de tamaño n-by-(n+1), donde n es el número de incógnitas.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=np.float32)
    b = np.array(b, dtype=float).reshape(-1, 1)
    Ab = np.hstack((A,b))
    assert A.shape[0] == Ab.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    
    n = Ab.shape[0]
    print(Ab)
    for i in range(n-1,0,-1): 
        # --- Eliminación: loop por fila
        for j in range(i, 0,-1):
            if Ab[j-1, i] != 0:
                m = Ab[j-1, i]/Ab[i,i]
                Ab[j-1, i:] = Ab[j-1, i:] - m * Ab[i, i:]
            print(f"\n{Ab}")
    if Ab[n - 1, n - 1] == 0:
        print("\nNo existe solución.")
        return None

    # --- Sustitución hacia atrás
    solucion = np.zeros(n, dtype=np.float32)
    solucion[n - 1] = Ab[n - 1, n] / Ab[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += Ab[i, j] * solucion[j]
        solucion[i] = (Ab[i, n] - suma) / Ab[i, i]

    return solucion