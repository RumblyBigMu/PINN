import torch
from torch import nn
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

R = 1.  # максимальный темп роста популяции
T0 = 0.  # начальное время
F0 = 1.  # граничное условие


def logistic_equation(t, y):
    """
    Логистическое уравнение (уравнение популяции)

    :param t: время
    :return: значение популяции в момент времени t
    """
    return R * t * (1 - t)


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

        self.layer_in = nn.Linear(1, size_hidden)  # 1 входное значение
        self.layer_out = nn.Linear(size_hidden, 1)  # 1 выходное значение

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList([nn.Linear(size_hidden, size_hidden) for _ in range(num_middle)])
        self.act = act  # функция активации

    def forward(self, x):
        """
        Функция распространения

        :param x: входное значение x
        :return: выходное значение
        """
        out = self.act(self.layer_in(x))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)


def f(nn, x):
    """
    Значение приближенного решения NN

    :param nn: экземпляр нейронной сети (NN)
    :param x: значение  переменной x
    :return: значение приближенного решения
    """
    return nn(x)


def df(nn, x=None, order=1):
    """
    Значение производной

    :param nn: экземпляр нейронной сети (NN)
    :param x: значение  переменной x
    :param order: порядок производной
    :return: значение прозводной
    """
    df_value = f(nn, x)
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            x,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_value


def compute_loss(nn, x=None, verbose=False):
    """
    Функция потерь: L = L_de + L_bc

    :param nn: экземпляр нейронной сети (NN)
    :param x: значение  переменной x
    :param verbose: параметр
    :return: значение потерь
    """
    # Внутренние потери
    interior_loss = df(nn, x) - R * x * (1 - x)
    # Потери на граничном условии
    boundary = torch.Tensor([T0])
    boundary.requires_grad = True
    boundary_loss = f(nn, boundary) - F0

    final_loss = interior_loss.pow(2).mean() + boundary_loss ** 2
    return final_loss


def train_model(nn, loss_function, learning_rate, max_epochs):
    """
    Тренировка нейронной сети

    :param nn: экземпляр нейронной сети (NN)
    :param loss_function: функция потерь
    :param learning_rate: скорость обучения
    :param max_epochs: число эпох
    :return: экземпляр сети, массив значений loss
    """
    loss_evolution = []
    # Оптимизация путем стохастического градиентного спуска
    optimizer = torch.optim.SGD(nn.parameters(), lr=learning_rate)
    # Главный тренировочный цикл
    for epoch in range(max_epochs):
        loss = loss_function(nn)  # функция потерь
        optimizer.zero_grad()  # обнуление (перезапуск) градиентов
        loss.backward()  # обратное распространение ошибки
        optimizer.step()  # шаг оптимизатора

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} -> Loss: {float(loss):>7f}")

        loss_evolution.append(loss.detach().numpy())
    return nn, np.array(loss_evolution)


def plot_solution(numeric_solution, net_solution, t_eval):
    fig, ax = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
    ax[0].plot(t_eval, net_solution, label="NN solution")
    ax[0].plot(t_eval, numeric_solution, label=f"Analytic solution", color="green")
    ax[0].set(title="Logistic equation", xlabel="t", ylabel="f(t)")
    ax[0].legend()

    ax[1].semilogy(loss_evolution)
    ax[1].set(title="Loss evolution", xlabel="Epoch", ylabel="Loss")

    plt.show()


def error(numeric_solution, net_solution):
    error = np.fabs(numeric_solution - net_solution)
    print(f"Error: {np.max(error)}")


if __name__ == "__main__":
    N = 100
    t_interval = [T0, 1.0]
    x = torch.linspace(t_interval[0], t_interval[1], steps=N, requires_grad=True)
    x = x.reshape(x.shape[0], 1)

    # Инициализация нейронной сети
    net = Net(4, 10)
    print("Архитектура сети: \n", net, end='\n\n')
    # Тренировка нейронной сети
    loss_function = partial(compute_loss, x=x, verbose=True)
    net_trained, loss_evolution = train_model(net, loss_function=loss_function, learning_rate=0.1, max_epochs=20000)

    # Численное решение
    M = 100
    t_eval = torch.linspace(t_interval[0], t_interval[1], steps=M).reshape(-1, 1)
    numeric_solution = solve_ivp(logistic_equation, t_interval, [F0], t_eval=t_eval.squeeze().detach().numpy())

    # Решение, полученное тренировкой сети
    net_solution = f(net_trained, t_eval)

    # Отрисовка графиков
    error(numeric_solution.y.T, net_solution.detach().numpy())
    plot_solution(numeric_solution.y.T, net_solution.detach().numpy(), t_eval.detach().numpy())
