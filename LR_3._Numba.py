import numpy as np
import timeit
from numba import njit



def MatyasFunc(x, y):
    return 0.26 * ((x ** 2) + (y ** 2)) - 0.48 * x * y

def GradMatyasFunc(x, y):
    return 0.52 * x - 0.48 * y, -0.48 * x + 0.52 * y

def McCormickFunc(x, y):
    return np.sin(x + y) + ((x - y) ** 2) - 1.5 * x + 2.5 * y + 1

def GradMcCormickFunc(x, y):
    part1 = 2 * x - 2 * y + np.cos(x + y) - 1.5
    part2 = 2 * y - 2 * x + np.cos(x + y) + 2.5
    return part1, part2 


@njit #alias for @numba.jit(nopython = True)
def numba_MatyasFunc(x, y):
    return 0.26 * ((x ** 2) + (y ** 2)) - 0.48 * x * y

@njit
def numba_GradMatyasFunc(x, y):
    return 0.52 * x - 0.48 * y, -0.48 * x + 0.52 * y

@njit
def numba_McCormickFunc(x, y):
    return np.sin(x + y) + ((x - y) ** 2) - 1.5 * x + 2.5 * y + 1

@njit
def numba_GradMcCormickFunc(x, y):
    part1 = 2 * x - 2 * y + np.cos(x + y) - 1.5
    part2 = 2 * y - 2 * x + np.cos(x + y) + 2.5
    return part1, part2 


def GradL(func, g_func, theta0_i, theta1_i, alpha = 0.01, eps = 0.0000001):

    i = 0
    (theta0_i_new, theta1_i_new) = (1000, 1000)

    while (abs(func(theta0_i, theta1_i) - func(theta0_i_new, theta1_i_new)) >= eps):

        if i > 0:
            theta0_i = theta0_i_new
            theta1_i = theta1_i_new

        i += 1

        upd_theta0_i, upd_theta1_i = g_func(theta0_i, theta1_i)
        theta0_i_new = theta0_i - alpha * upd_theta0_i
        theta1_i_new = theta1_i - alpha * upd_theta1_i

    return i, theta0_i, theta1_i, func(theta0_i, theta1_i)


@njit #alias for @numba.jit(nopython = True)
def numba_GradL(func, g_func, theta0_i, theta1_i, alpha = 0.01, eps = 0.0000001):

    i = 0
    (theta0_i_new, theta1_i_new) = (1000, 1000)

    while (abs(func(theta0_i, theta1_i) - func(theta0_i_new, theta1_i_new)) >= eps):

        if i > 0:
            theta0_i = theta0_i_new
            theta1_i = theta1_i_new

        i += 1

        upd_theta0_i, upd_theta1_i = g_func(theta0_i, theta1_i)
        theta0_i_new = theta0_i - alpha * upd_theta0_i
        theta1_i_new = theta1_i - alpha * upd_theta1_i

    return i, theta0_i, theta1_i, func(theta0_i, theta1_i)

x, y = 5, 10

print('\n           Matyas function\n')

setup_code = """
from __main__ import GradL, MatyasFunc, GradMatyasFunc, x, y
func = MatyasFunc
g_func = GradMatyasFunc
theta0_i = x
theta1_i = y
"""
print('Time without Numba = ', timeit.timeit(stmt = "GradL(func, g_func, theta0_i, theta1_i)", 
                                             setup = setup_code, 
                                             number = 100))

#stepsGD, x_resGD, y_resGD, resGD = numba_GradL(numba_MatyasFunc, numba_GradMatyasFunc, x, y)
numba_setup_code = """
from __main__ import numba_GradL, numba_MatyasFunc, numba_GradMatyasFunc, x, y
func = numba_MatyasFunc
g_func = numba_GradMatyasFunc
theta0_i = x
theta1_i = y
"""
print('Time with Numba = ', timeit.timeit(stmt = "numba_GradL(func, g_func, theta0_i, theta1_i)", 
                                          setup = numba_setup_code, 
                                          number = 100))

stepsGD, x_resGD, y_resGD, resGD = GradL(MatyasFunc, GradMatyasFunc, x, y)
print('\nGradient descent X result = ' + str(x_resGD) + '\n' + 
      'Gradient descent Y result = ' + str(y_resGD) + '\n' + 
      'Gradient descent function result = ' + str(resGD) + '\n' +
      'Epochs = ' + str(stepsGD) + '\n')


x, y = -1, 3
 
print('\n           McCormick function\n')

setup_code = """
from __main__ import GradL, McCormickFunc, GradMcCormickFunc, x, y
func = McCormickFunc
g_func = GradMcCormickFunc
theta0_i = x
theta1_i = y
"""
print('Time without Numba = ', timeit.timeit(stmt = "GradL(func, g_func, theta0_i, theta1_i)", 
                                             setup = setup_code, 
                                             number = 1000))

#stepsGD, x_resGD, y_resGD, resGD = numba_GradL(numba_McCormickFunc, numba_GradMcCormickFunc, x, y)
numba_setup_code = """
from __main__ import numba_GradL, numba_McCormickFunc, numba_GradMcCormickFunc, x, y
func = numba_McCormickFunc
g_func = numba_GradMcCormickFunc
theta0_i = x
theta1_i = y
"""
print('Time with Numba = ', timeit.timeit(stmt = "numba_GradL(func, g_func, theta0_i, theta1_i)", 
                                          setup = numba_setup_code, 
                                          number = 1000))

stepsGD, x_resGD, y_resGD, resGD = GradL(McCormickFunc, GradMcCormickFunc, x, y)
print('\nGradient descent X result = ' + str(x_resGD) + '\n' + 
      'Gradient descent Y result = ' + str(y_resGD) + '\n' + 
      'Gradient descent function result = ' + str(resGD) + '\n' +
      'Epochs = ' + str(stepsGD) + '\n')