import optuna
import numpy as np
import plotly


def MatyasFunc(x, y):
    return 0.26 * ((x ** 2) + (y ** 2)) - 0.48 * x * y


def McCormickFunc(x, y):
    return np.sin(x + y) + ((x - y) ** 2) - 1.5 * x + 2.5 * y + 1          

def objective(trial, func, x_low, x_high, y_low, y_high):
    x = trial.suggest_float("x", x_low, x_high)
    y = trial.suggest_float("y", y_low, y_high)
    return func(x, y)

def op_print(study):
    print("Best x, y values: ", study.best_params)
    print("Best function value: ", study.best_value)

def draw(study):
    fig = optuna.visualization.plot_contour(study)
    fig.show()

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()


study_matyas = optuna.create_study(study_name = "x=(-10,10), y=(-10,10)")
study_matyas.optimize(lambda t: objective(t, MatyasFunc, -10, 10, -10, 10), n_trials=300, show_progress_bar = True)

study_mccormick = optuna.create_study(study_name = "x=(-1.5,4), y=(-3,4)")
study_mccormick.optimize(lambda t: objective(t, McCormickFunc, -1.5, 4, -3, 4), n_trials=300, show_progress_bar = True)


print('\n           Matyas function\n')
op_print(study_matyas)
draw(study_matyas)
 

print('\n           McCormick function\n')
op_print(study_mccormick)
draw(study_mccormick)


fig = optuna.visualization.plot_edf([study_matyas, study_mccormick])
fig.show()