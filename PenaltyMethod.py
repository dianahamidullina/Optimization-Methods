import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


class CONSTANTS:
    eps = 1e-4  # точность
    r_k = 1  # начальное значение параметра штрафа
    C = 8  # для увеличения штрафа
    alpha = 0.1  # для метода линии поиска
    beta = 0.8  # для метода линии поиска


def gradient(func, x, y, h=1e-4):
    """Вычисление градиента функции"""
    df_dx = (func(x + h, y) - func(x - h, y)) / (2 * h)
    df_dy = (func(x, y + h) - func(x, y - h)) / (2 * h)
    return df_dx, df_dy


def hessian(func, x, y, h=1e-4):
    """Вычисление матрицы Гессе"""
    df_dx2 = (func(x + h, y) - 2 * func(x, y) + func(x - h, y)) / (h ** 2)
    df_dy2 = (func(x, y + h) - 2 * func(x, y) + func(x, y - h)) / (h ** 2)
    df_dxdy = (func(x + h, y + h) - func(x + h, y - h) - func(x - h, y + h) + func(x - h, y - h)) / (4 * h ** 2)
    return [[df_dx2, df_dxdy], [df_dxdy, df_dy2]]


def solve_linear_system(A, b):
    """Решение системы линейных уравнений Ax = b"""
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    inv_A = [[A[1][1] / det, -A[0][1] / det], [-A[1][0] / det, A[0][0] / det]]
    return [inv_A[0][0] * b[0] + inv_A[0][1] * b[1], inv_A[1][0] * b[0] + inv_A[1][1] * b[1]]


def newton_method(func, x):
    """Метод Ньютона для нахождения экстремума"""
    x_k = x[:]
    itr = 0
    while itr < 100:  # Ограничиваем количество итераций
        grad_x, grad_y = gradient(func, x_k[0], x_k[1])
        hess = hessian(func, x_k[0], x_k[1])
        step = solve_linear_system(hess, [grad_x, grad_y])
        x_next = [x_k[0] - step[0], x_k[1] - step[1]]

        if abs(func(x_next[0], x_next[1]) - func(x_k[0], x_k[1])) <= CONSTANTS.eps:
            return x_next
        x_k = x_next
        itr += 1
    return [0, 0]


def f(x, y):
    """Целевая функция"""
    return 9 * (x - 5) ** 2 + 4 * (y - 5) ** 2


def g1(x, y):
    """Ограничения"""
    return x ** 2 - 2 * x - y + 1


def g2(x, y):
    return -x + y - 1


def g3(x, y):
    return x - y


def g_star(g):
    """Квадрат срезки для штрафной функции"""
    return max(0, g) ** 2


def penalty_func(x, y):
    """Функция штрафа для ограничений"""
    constraints = [g1(x, y), g2(x, y), g3(x, y)]
    penalty = sum(g_star(g) for g in constraints[:3])  # Штраф для j = 1..m
    return penalty


def helper_func(x, y):
    """Функция с учетом штрафа"""
    constraints = [g1(x, y), g2(x, y), g3(x, y)]
    penalty = (CONSTANTS.r_k / 2) * sum(g_star(g) for g in constraints[:3])
    return f(x, y) + penalty


def calculate():
    """Основная функция для вычисления оптимальной точки"""
    k = 0
    x_dash_k = [0.0, 0.0]  # начальная точка

    results = []

    while True:
        x_next = newton_method(helper_func, x_dash_k)

        results.append([k + 1, x_dash_k[0], x_dash_k[1], x_next[0], x_next[1], CONSTANTS.r_k])

        x_dash_k = x_next

        if penalty_func(x_dash_k[0], x_dash_k[1]) <= CONSTANTS.eps:
            print("\nТаблица результатов:")
            print(tabulate(results, headers=["Итерация", "x_dash_k[0]", "x_dash_k[1]", "x_next[0]", "x_next[1]", "r_k"],
                           tablefmt="grid"))
            print(f"\nОкончательная точка: {x_dash_k[0]:8.4f} {x_dash_k[1]:8.4f}")
            print(f"Количество итераций: {k + 1}")

            # Добавляем график
            plot_constraints(x_dash_k)  # Передаем оптимальную точку в функцию plot_constraints
            return
        else:
            CONSTANTS.r_k *= CONSTANTS.C
            k += 1


def plot_constraints(optimal_point):
    """Функция для построения графика области допустимых значений (ОДЗ)"""
    x = np.linspace(-1, 6, 400)
    y = np.linspace(-1, 6, 400)
    X, Y = np.meshgrid(x, y)

    # Ограничения
    Z1 = g1(X, Y)
    Z2 = g2(X, Y)
    Z3 = g3(X, Y)

    plt.figure(figsize=(8, 6))

    # Отображаем области допустимых значений (где g1(x, y) <= 0, g2(x, y) <= 0, g3(x, y) <= 0)
    plt.contour(X, Y, Z1, levels=[0], colors='r')
    plt.contour(X, Y, Z2, levels=[0], colors='g')
    plt.contour(X, Y, Z3, levels=[0], colors='b')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Область допустимых значений (ОДЗ) и оптимальная точка')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)

    # Отображаем решение
    plt.scatter([optimal_point[0]], [optimal_point[1]], color='purple', label="Оптимальная точка", zorder=5)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    calculate()
