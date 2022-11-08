import torch
from torch import nn
import numpy as np
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt

# Заданные параметры уравнения теплопроводности
k = 1. / (np.pi ** 2)
F = lambda t, x: (torch.cos(torch.pi * x) - x * torch.exp(-t))
Phi = lambda x: x
Psi_0 = lambda t: 1.
Psi_1 = lambda t: (2. * torch.exp(-t) - 1.)
# Точное решение
solution = lambda t, x: (1. - np.exp(-t)) * np.cos(np.pi * x) + x * np.exp(-t)


# Начальное условие: u(0,x)=Phi(x)
def initial_condition(x):
    res = Phi(x).detach().clone()
    return res


# Левое граничное условие: u(t,0)+du(t,0)/dx = Psi_0(t)
def boundary_condition_left(t):
    res = Psi_0(t)
    res = torch.tensor(res)
    return res


# Правое граничное условие: u(t, 1) = Psi_1(t)
def boundary_condition_right(t):
    res = Psi_1(t)
    return res


class Net(nn.Module):
    """
    Класс для построения нейронной сети
    """

    def __init__(self, num_hidden, size_hidden, act=nn.Tanh()):
        """
        Архитектура нейронной сети

        :param num_hidden: количество скрытых слоев
        :param size_hidden: количество скрытых нейронов
        :param act: функция активации
        """
        super().__init__()

        self.layer_in = nn.Linear(2, size_hidden)  # 2 входных значения
        self.layer_out = nn.Linear(size_hidden, 1)  # 1 выходное значение

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList([nn.Linear(size_hidden, size_hidden) for _ in range(num_middle)])
        self.act = act  # функция активации

    def forward(self, x, t):
        """
        Функция распространения

        :param x: входное значение x
        :param t: входное значение t
        :return: выходное значение
        """
        x_stack = torch.cat([x, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)


def f(nn, x, t):
    """
    Значение приближенного решения NN

    :param nn: экземпляр нейронной сети (NN)
    :param x: значение  переменной x
    :param t: значение  переменной t
    :return: значение приближенного решения
    """
    return nn(x, t)


def df(f_value, variable, order=1):
    """
    Значение производной

    :param f_value: значение функции
    :param variable: значение  переменной
    :param order: порядок производной
    :return: значение производной
    """
    df_value = f_value
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            variable,
            grad_outputs=torch.ones_like(variable),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


def dfdt(nn, x, t, order=1):
    """
    Производная по временной переменной произвольного порядка

    :param nn: экземпляр нейронной сети (NN)
    :param x: значение  переменной x
    :param t: значение  переменной t
    :param order: порядок производной
    :return: значение производной df/dt
    """
    f_value = f(nn, x, t)
    return df(f_value, t, order=order)


def dfdx(nn, x, t, order=1):
    """
    Производная по пространственной переменной произвольного порядка

    :param nn: экземпляр нейронной сети (NN)
    :param x: значение  переменной x
    :param t: значение  переменной t
    :param order: порядок производной
    :return: значение производной df/dx
    """
    f_value = f(nn, x, t)
    return df(f_value, x, order=order)


def compute_loss(nn, x=None, t=None):
    """
    Функция потерь: L = L_de + L_ic + L_bc

    :param nn: экземпляр нейронной сети (NN)
    :param x: переменная x
    :param t: переменная t
    :return: значение потерь
    """
    # Внутренние потери
    interior_loss = dfdt(nn, x, t, order=1) - k * dfdx(nn, x, t, order=2) - F(t, x)
    # Потери на граничных условиях
    t_raw = torch.unique(t).reshape(-1, 1).detach().numpy()
    t_raw = torch.Tensor(t_raw)
    t_raw.requires_grad = True
    boundary_left = torch.ones_like(t_raw, requires_grad=True) * x[0]
    boundary_loss_left = f(nn, boundary_left, t_raw) + dfdx(nn, boundary_left, t_raw,
                                                            order=1) - boundary_condition_left(boundary_left)

    boundary_right = torch.ones_like(t_raw, requires_grad=True) * x[-1]
    boundary_loss_right = f(nn, boundary_right, t_raw) - boundary_condition_right(t_raw)

    boundary_loss = boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean()
    # Потери на начальном условии
    x_raw = torch.unique(x).reshape(-1, 1).detach().numpy()
    x_raw = torch.Tensor(x_raw)
    x_raw.requires_grad = True

    f_initial = initial_condition(x_raw)
    t_initial = torch.zeros_like(x_raw)
    initial_loss = f(nn, x_raw, t_initial) - f_initial

    final_loss = interior_loss.pow(2).mean() + initial_loss.pow(2).mean() + boundary_loss

    return final_loss


def train_model(nn, loss_function, learning_rate=0.01, max_epochs=1000):
    """
    Тренировка нейронной сети

    :param nn: экземпляр нейронной сети (NN)
    :param loss_function: функция потерь
    :param learning_rate: скорость обучения
    :param max_epochs: число эпох
    :return: экземпляр сети, массив значений loss
    """
    start_time = datetime.now()
    loss_evolution = []
    # Оптимизация путем оценки адаптивного момента (Adam)
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate)
    # Главный тренировочный цикл
    for epoch in range(max_epochs):
        loss = loss_function(nn)  # функция потерь
        optimizer.zero_grad()  # обнуление (перезапуск) градиентов
        loss.backward()  # обратное распространение ошибки
        optimizer.step()  # шаг оптимизатора

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} -> Loss: {float(loss):>7f}")

        loss_evolution.append(loss.detach().numpy())
    print("Время тренировки: ", datetime.now() - start_time, "\n")
    return nn, np.array(loss_evolution)


def plot_solution(analytic_solution, net_solution, x_eval, t_eval, flag=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.semilogy(loss_evolution)
    ax.set(title="Loss evolution", xlabel="Epoch", ylabel="Loss")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    x_axis, t_axis = np.meshgrid(x_eval, t_eval)
    if flag:
        ax.plot_surface(x_axis, t_axis, analytic_solution)
        ax.plot_surface(x_axis, t_axis, net_solution)
    else:
        ax.plot_surface(x_axis, t_axis, error)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')

    plt.show()


def error(analytic_solution, net_solution):
    error = np.fabs(analytic_solution - net_solution)
    print(f"Error: {np.max(error)}")


if __name__ == "__main__":
    T = 1.
    x_interval = [0.0, 1.]
    N_x = 100
    t_interval = [0.0, T]
    N_t = 100

    x_raw = torch.linspace(x_interval[0], x_interval[1], steps=N_x, requires_grad=True)
    t_raw = torch.linspace(t_interval[0], t_interval[1], steps=N_t, requires_grad=True)
    grids = torch.meshgrid(x_raw, t_raw, indexing='ij')

    x = grids[0].flatten().reshape(-1, 1)
    t = grids[1].flatten().reshape(-1, 1)

    # Инициализация нейронной сети
    net = Net(3, 20)
    print("Архитектура сети: \n", net, end='\n\n')
    # Тренировка нейронной сети
    loss_function = partial(compute_loss, x=x, t=t)
    net_trained, loss_evolution = train_model(net, loss_function=loss_function, learning_rate=0.001, max_epochs=50000)

    x_eval = torch.linspace(x_interval[0], x_interval[1], steps=N_x).reshape(-1, 1)
    t_eval = torch.linspace(t_interval[0], t_interval[1], steps=N_t).reshape(-1, 1)


    def analyticSolution(T, N, M):
        """
        Получение аналитического решения

        :param T: время
        :param N: количество точек по пространству
        :param M: количество точек по времени
        :return: значение функции u
        """
        h = 1 / N
        tau = T / M
        U = np.zeros((M + 1, N + 1))
        for j in range(M + 1):  # цикл по времени
            for i in range(N + 1):  # цикл по пространству
                U[j][i] = solution(j * tau, i * h)
        return U


    def netSolution(T, N, M):
        """
        Получение решения сети

        :param T: время
        :param N: количество точек по пространству
        :param M: количество точек по времени
        :return: значение функции u
        """
        h = 1 / N
        tau = T / M
        U = np.zeros((M + 1, N + 1))
        for j in range(M + 1):  # цикл по времени
            for i in range(N + 1):  # цикл по пространству
                U[j][i] = f(net_trained, torch.tensor(i * h).reshape(-1, 1), torch.tensor(j * tau).reshape(-1, 1))
        return U

    # Аналитическое решение
    analytic_solution = analyticSolution(T, N_x - 1, N_t - 1)
    # Решение, полученное тренировкой сети
    net_solution = netSolution(T, N_x - 1, N_t - 1)

    error(analytic_solution, net_solution)
    plot_solution(analytic_solution, net_solution, x_eval, t_eval, flag=False)
