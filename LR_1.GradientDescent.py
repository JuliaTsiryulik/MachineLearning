import numpy as np
import torch
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, cm
from sklearn.datasets import make_blobs
from matplotlib.animation import PillowWriter
import celluloid
from celluloid import Camera
import pandas as pd
#from IPython.display import HTML, Image

#Loss function
def GradL(func, g_func, theta0_i, theta1_i, alpha = 0.01, eps = 0.0000001):

    theta0_i_list = []
    theta1_i_list = []
    func_list = []

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

        theta0_i_list.append(theta0_i_new)
        theta1_i_list.append(theta1_i_new)
        func_list.append(func(theta0_i_new, theta1_i_new))


    return i, theta0_i, theta1_i, func(theta0_i, theta1_i), theta0_i_list, theta1_i_list, func_list


def GD_lr_schedule(func, g_func, theta0_i, theta1_i, start_lr = 0.1, drop = 0.5, rate = 10000, eps = 0.0000001):

    i = 0
    (theta0_i_new, theta1_i_new) = (1000, 1000)

    while (abs(func(theta0_i, theta1_i) - func(theta0_i_new, theta1_i_new)) >= eps):

        if i > 0:
            theta0_i = theta0_i_new
            theta1_i = theta1_i_new

        i += 1

        #if (i != 0) and (i % (1000 / 10) == 0):
         #   alpha *= 0.1

        lr = start_lr * drop ** ((1 + i) / rate)

        upd_theta0_i, upd_theta1_i = g_func(theta0_i, theta1_i)
        theta0_i_new = theta0_i - lr * upd_theta0_i
        theta1_i_new = theta1_i - lr * upd_theta1_i

    return i, theta0_i, theta1_i, func(theta0_i, theta1_i)


"""
def SGD(func, g_func, theta0_i, theta1_i, alpha = 0.1):

    #x_data = torch.normal(mean = torch.arange(0.0, 1.), std = torch.arange(1, 0, -0.1))
    #y_data = torch.normal(mean = torch.arange(0.0, 1.), std = torch.arange(1, 0, -0.1))
    #x_data, y_data = make_blobs(n_samples = 10, n_features = 1, random_state = 0, center_box = (0.0, 1.0))

    for i in range(1100):
      #  for x, y in zip(x_data.numpy(), y_data.numpy()):
            #upd_theta0_i, upd_theta1_i = g_func(theta0_i, theta1_i)
       #     upd_theta0_i, upd_theta1_i = g_func(x, y)

        #    theta0_i = theta0_i - alpha * upd_theta0_i
         #   theta1_i = theta1_i - alpha * upd_theta1_i


            upd_theta0_i = upd_theta0_i + torch.normal(0.0, 1, (1,))
            upd_theta1_i = upd_theta1_i + torch.normal(0.0, 1, (1,))

            theta0_i = theta0_i - alpha * upd_theta0_i.numpy()[0]
            theta1_i = theta1_i - alpha * upd_theta1_i.numpy()[0]

    return theta0_i, theta1_i, func(x, y)


def SGD_lr_schedule(func, g_func, theta0_i, theta1_i, alpha = 0.01):
    for i in range(1100000):

        if (i != 0) and (i % (100000 / 10) == 0):
            alpha *= 0.1
        
        upd_theta0_i, upd_theta1_i = g_func(theta0_i, theta1_i)

        upd_theta0_i = upd_theta0_i + torch.normal(0.0, 1, (1,))
        upd_theta1_i = upd_theta1_i + torch.normal(0.0, 1, (1,))

        theta0_i = theta0_i - alpha * upd_theta0_i.numpy()[0]
        theta1_i = theta1_i - alpha * upd_theta1_i.numpy()[0]

    return theta0_i, theta1_i, func(theta0_i, theta1_i)
"""

def Nesterov_momentum(func, g_func, theta0_i, theta1_i, alpha = 0.001, gamma = 0.9, eps = 0.0000001):
    v0, v1 = 0, 0

    i = 0
    (theta0_i_new, theta1_i_new) = (1000, 1000)

    while (abs(func(theta0_i, theta1_i) - func(theta0_i_new, theta1_i_new)) >= eps):

        if i > 0:
            theta0_i = theta0_i_new
            theta1_i = theta1_i_new

        i += 1

        upd_theta0_i, upd_theta1_i = g_func(theta0_i, theta1_i)

        v0 = gamma * v0 + alpha * upd_theta0_i
        v1 = gamma * v1 + alpha * upd_theta1_i

        theta0_i_new = theta0_i - v0
        theta1_i_new = theta1_i - v1

    return i, theta0_i, theta1_i, func(theta0_i, theta1_i)


def Adam_momentum(func, g_func, theta0_i, theta1_i, alpha = 0.01, beta0 = 0.9, beta1 = 0.999, eps = 10e-8, t = 9, epsilon = 0.0000001):
    first_moment0, first_moment1 = 0, 0
    second_moment0, second_moment1 = 0, 0

    i = 0
    (theta0_i_new, theta1_i_new) = (1000, 1000)

    while (abs(func(theta0_i, theta1_i) - func(theta0_i_new, theta1_i_new)) >= epsilon):

        if i > 0:
            theta0_i = theta0_i_new
            theta1_i = theta1_i_new

        i += 1

        upd_theta0_i, upd_theta1_i = g_func(theta0_i, theta1_i)

        first_moment0 = beta0 * first_moment0 + (1 - beta0) * upd_theta0_i
        first_moment1 = beta0 * first_moment1 + (1 - beta0) * upd_theta1_i

        second_moment0 = beta1 * second_moment0 + (1 - beta1) * (upd_theta0_i ** 2)
        second_moment1 = beta1 * second_moment1 + (1 - beta1) * (upd_theta1_i ** 2)

        first_unbias0 = first_moment0 / (1 - beta0 ** t)
        first_unbias1 = first_moment1 / (1 - beta0 ** t)

        second_unbias0 = second_moment0 / (1 - beta1 ** t)
        second_unbias1 = second_moment1 / (1 - beta1 ** t)

        theta0_i_new = theta0_i - ((alpha * first_unbias0) / np.sqrt(second_unbias0 + eps))
        theta1_i_new = theta1_i - ((alpha * first_unbias1) / np.sqrt(second_unbias1 + eps))

    return i, theta0_i, theta1_i, func(theta0_i, theta1_i)


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
           

def draw(func, start_x, stop_x, start_y, stop_y, title_name, x_list, y_list, z_list, gif_name, steps):
    

    # configure full path for ImageMagick
    #rcParams['animation.convert_path'] = r'/usr/bin/convert'


    x = np.linspace(start_x, stop_x, 30)
    y = np.linspace(start_y, stop_y, 30)

    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)


    # Introduce training data
    x_train = np.array([     
        [1],
        [2],
        [4],
        [5],
        [6],
        [7]
    ])

    y_train = np.array([     
        [4],
        [-12],
        [3],
        [-11],
        [-5],
        [-17]
    ])


    zs = np.array([cost_3d(x_train, y_train,       # determine costs for each pair of w and b 
                   np.array([[x]]), np.array([[y]]))  # cost_3d() only accepts wp and bp as matrices. 
                   for x, y in zip(np.ravel(X), np.ravel(Y))])

    Z = Z.reshape(X.shape)

    #fig = plt.figure(figsize=(10,10))
    
    fig = plt.figure()
    camera = Camera(fig)
    #ax = plt.axes(projection = '3d')
    #ax = p3.Axes3D(fig)




    # Define which epochs/data points to plot
    a = np.arange(0, 50, 1).tolist()
    b = np.arange(50, 100, 5).tolist()
    c = np.arange(100, steps, 200).tolist()
    points = a + b + c # points we want to plot




    # Turn lists into arrays
    """
    x_list = np.array(x_list).flatten()
    y_list = np.array(y_list).flatten()
    z_list = np.array(z_list).flatten() 
    """
    points = np.array(points)

   

    ax = fig.add_subplot(projection = '3d') # projection='3d'
    ax.set_title(title_name, fontsize = 30)

    
    ax.set_xlabel("x", fontsize=25, labelpad=10)
    ax.set_ylabel("y", fontsize=25, labelpad=10)
    ax.set_zlabel("z", fontsize=25, labelpad=-35) # negative value for labelpad places z-label left of z-axis.

    bb = 0 
    for i in points: 
        if bb > 0: 
            bb = bb - 1 
            continue 
        bb = 10

        #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='b', alpha=0.35) # create surface plot
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap = cm.coolwarm, alpha = 0.35) # create surface plot
        ax.scatter(x_list[i], y_list[i], z_list[i], marker='o', s=12**2, color='red' )
        ax.tick_params(axis='both', which='major', labelsize=15) 
        leg = ax.plot(x_list[0:i], y_list[0:i], z_list[0:i], linestyle="dashed", linewidth=2, color="grey") # (dashed) lineplot

        ax.legend(leg,[f'Min = {z_list[i]}'], loc='upper right', fontsize=15)
    
        plt.tight_layout()
        camera.snap()
    
    #ax.view_init(30, -45)
    anim = camera.animate(interval = 5, repeat = False, repeat_delay = 500)

    #plt.show()


    anim.save(gif_name, writer = 'pillow')
    #HTML(anim.to_jshtml())
    



def cost_3d(x, y, w, b):  # predicts costs for every pair of w and b. 
    pred = x@w.T + b                       
    e = y - pred
    return np.mean(e ** 2)

#x, y = make_blobs(n_samples = 100, n_features = 1, random_state = 0)
x, y = 5, 10


print('Matyas function\n')



stepsGD, x_resGD, y_resGD, resGD, x_listGD, y_listGD, z_listGD = GradL(MatyasFunc, GradMatyasFunc, x, y)
draw(MatyasFunc, -10, 10, -10, 10, 'Matyas function', x_listGD, y_listGD, z_listGD, 'Matyas_function.gif', stepsGD)
print('Gradient descent X result = ' + str(x_resGD) + '\n' + 
      'Gradient descent Y result = ' + str(y_resGD) + '\n' + 
      'Gradient descent function result = ' + str(resGD) + '\n' +
      'Epochs = ' + str(stepsGD) + '\n')

stepsGD_lr, x_resGD_lr, y_resGD_lr, resGD_lr = GD_lr_schedule(MatyasFunc, GradMatyasFunc, x, y)
print('Gradient descent and learning rate schedule X result = ' + str(x_resGD_lr) + '\n' + 
      'Gradient descent and learning rate schedule Y result = ' + str(y_resGD_lr) + '\n' + 
      'Gradient descent and learning rate schedule function result = ' + str(resGD_lr) + '\n' +
      'Epochs = ' + str(stepsGD_lr) + '\n')

"""
x_resSGD, y_resSGD, resSGD = SGD(MatyasFunc, GradMatyasFunc, x, y)
print('Stochastic gradient descent X result = ' + str(x_resSGD) + '\n' + 
      'Stochastic gradient descent Y result = ' + str(y_resSGD) + '\n' + 
      'Stochastic gradient descent function result = ' + str(resSGD) + '\n')

x_resSGD_lr, y_resSGD_lr, resSGD_lr = SGD_lr_schedule(MatyasFunc, GradMatyasFunc, x, y)
print('Stochastic gradient descent and learning rate schedule X result = ' + str(x_resSGD_lr) + '\n' + 
      'Stochastic gradient descent and learning rate schedule Y result = ' + str(y_resSGD_lr) + '\n' + 
      'Stochastic gradient descent and learning rate schedule function result = ' + str(resSGD_lr) + '\n')
"""

stepsNes, x_Nes, y_Nes, resNes = Nesterov_momentum(MatyasFunc, GradMatyasFunc, x, y)
#draw(MatyasFunc, -10, 10, -10, 10, 'Matyas function', x_Nes, y_Nes, resNes, 'Matyas_function.gif')
print('Nesterov momentum X result = ' + str(x_Nes) + '\n' + 
      'Nesterov momentum Y result = ' + str(y_Nes) + '\n' + 
      'Nesterov momentum function result = ' + str(resNes) + '\n' +
      'Epochs = ' + str(stepsNes) + '\n')

stepsAdam, x_Adam, y_Adam, resAdam = Adam_momentum(MatyasFunc, GradMatyasFunc, x, y)
print('Adam momentum X result = ' + str(x_Adam) + '\n' + 
      'Adam momentum Y result = ' + str(y_Adam) + '\n' + 
      'Adam momentum function result = ' + str(resAdam) + '\n' +
      'Epochs = ' + str(stepsAdam) + '\n')


x, y = -1, 3
 
print('McCormick function\n')


stepsGD, x_resGD, y_resGD, resGD, x_listGD, y_listGD, z_listGD = GradL(McCormickFunc, GradMcCormickFunc, x, y)
draw(McCormickFunc, -1.5, 4, -3, 4, 'McCormick function', x_listGD, y_listGD, z_listGD, 'McCormick_function.gif', stepsGD)
print('Gradient descent X result = ' + str(x_resGD) + '\n' + 
      'Gradient descent Y result = ' + str(y_resGD) + '\n' + 
      'Gradient descent function result = ' + str(resGD) + '\n' +
      'Epochs = ' + str(stepsGD) + '\n')

stepsGD_lr, x_resGD_lr, y_resGD_lr, resGD_lr = GD_lr_schedule(McCormickFunc, GradMcCormickFunc, x, y)
print('Gradient descent and learning rate schedule X result = ' + str(x_resGD_lr) + '\n' + 
      'Gradient descent and learning rate schedule Y result = ' + str(y_resGD_lr) + '\n' + 
      'Gradient descent and learning rate schedule function result = ' + str(resGD_lr) + '\n' +
      'Epochs = ' + str(stepsGD_lr) + '\n')

"""
x_resSGD, y_resSGD, resSGD = SGD(McCormickFunc, GradMcCormickFunc, x, y)
print('Stochastic gradient descent X result = ' + str(x_resSGD) + '\n' + 
      'Stochastic gradient descent Y result = ' + str(y_resSGD) + '\n' + 
      'Stochastic gradient descent function result = ' + str(resSGD) + '\n')

x_resSGD_lr, y_resSGD_lr, resSGD_lr = SGD_lr_schedule(McCormickFunc, GradMcCormickFunc, x, y)
print('Stochastic gradient descent and learning rate schedule X result = ' + str(x_resSGD_lr) + '\n' + 
      'Stochastic gradient descent and learning rate schedule Y result = ' + str(y_resSGD_lr) + '\n' + 
      'Stochastic gradient descent and learning rate schedule function result = ' + str(resSGD_lr) + '\n')
"""

stepsNes, x_Nes, y_Nes, resNes = Nesterov_momentum(McCormickFunc, GradMcCormickFunc, x, y)
print('Nesterov momentum X result = ' + str(x_Nes) + '\n' + 
      'Nesterov momentum Y result = ' + str(y_Nes) + '\n' + 
      'Nesterov momentum function result = ' + str(resNes) + '\n' +
      'Epochs = ' + str(stepsNes) + '\n')

stepsAdam, x_Adam, y_Adam, resAdam = Adam_momentum(McCormickFunc, GradMcCormickFunc, x, y)
print('Adam momentum X result = ' + str(x_Adam) + '\n' + 
      'Adam momentum Y result = ' + str(y_Adam) + '\n' + 
      'Adam momentum function result = ' + str(resAdam) + '\n' +
      'Epochs = ' + str(stepsAdam) + '\n')