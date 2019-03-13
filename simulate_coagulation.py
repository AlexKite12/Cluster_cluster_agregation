#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import List

@dataclass
class Particle:
    mass : float
    not_empty : bool = field(default=True)
    velocity : float = field(init=False)

    def make_collision(self, other):
        self.mass += other.mass

@dataclass
class Cluster(Particle):
    particles : List[Particle] = field(default_factory=list)
    number_particles : int = field(init = False)

"""
V(N) - объем системы из N частиц
m[k] = m[0] * k; m[0] = 1
u[k](t) - концентрация частиц массы k момент времени t
Ph[i, j] - интенсивность коагуляции частиц с энергиями i и j
S - оператор Смолуховского, задающий скорость изменения концентраций
S = sum(Ph[i, j] * u[i] * u[j]) - 2 * u[k] * sum(Ph[k, i] * u[i])
hs_function = lambda x : 0 if x < 0 else 1 - функция Хевисайда
M - искусственный предел размера частиц в системе
hM = hs_function(M - k)

Пусть для заданных уравнений рассматриается задача Коши с начальными данными:
u[k](0) = hM * phi[k]

Метод, основанного на случайном розыгрыше актов коагуляции на уровне отдельных частиц
Пусть :
    K - множество финитных функций

    Рассмотрим парную коагуляцию:
        Положим, что каждой частице присвоен номер i(1≤i≤N)
            и она имеет в момент времени t≥0 массу  m[i]
        Пусть V (N) = N
        Тогда состояние системы в момент времени t:
            m(t) = [m[1](t), m[2](t), ..., m[N](t)]
        Назовем коагуляцию пары частиц i, j (1 <= i < j <= N) преобразование
            A[i, j]:m->m`=A[i. j](m) такое, что m[k]->m[k], если k != i,j
                m[i]->m[i] + m[j]
                m[j]->0
            Паре сталкивающихся частиц с номерами i j , (i < j)
                в рассматриваемой системе взаимно однозначно сопоставим подстановку:
                substitution[i, j] = [[1, 2, ..., i, ..., j, ..., N],
                                        [1, 2, ..., i, ..., j, ..., N]]
                множество которых обозначим S2(N); card(S2(N)) = C2[N]
            Пусть пара сталкивающихся частиц разыгрывается в каждый момент времени:
                t[n] = n * dtau
                с вероятностью:
                choice_probability = 1 / C2[N]
            Определим случайную величину nu = random.choice([0, 1]),
                означающая коагуляцию или отсутствие оной.
            Положим, что вероятность коагуляции частицы энергии k с частицей энергии l равна:
                dtau(N)(N-1)Ph[k, l] <= 1
            Величину dtau назовем временем столкновения:
                0 <= dtau <= 1 / ((N - 1)||Ph2||)
"""
def calculate_step_time(Betta):
    return 2 / np.sum(Betta)

def choice_coagulation_pair(Betta, i, j):
    return random.choice([0, 1]) < Betta[i, j] / np.max(Betta)

if __name__ == '__main__':
    N = 1000 # number of particles
    Ph = np.empty((N, N))
    A = 0.1
    #Ph[:, :] = [[((i + 1) ** (0.75) * (j + 1) ** (-0.75) + (i + 1) ** (-0.75) * (j + 1) ** (0.75)) for j in range(N)] for i in range(N)]
    Ph[:, :] = [[ A * (i + j + 2) for j in range(N)] for i in range(N)]
    dtau = calculate_step_time(Ph)
    print(dtau)
    #t = np.arange(0.0, 1.0, dtau)
    particles_stack = np.array([Particle(1)] * N)

    fraction = []
    t = 2.0
    count = 0
    time = []
    while dtau * count < t:
    # for dt in range(len(t)):
        # while True:
        i = random.choice([i for i in range(N - 1)])
        j = random.choice([j for j in range(i, N)])
        if particles_stack[i].not_empty and particles_stack[j].not_empty:
            if choice_coagulation_pair(Ph, i, j):
                particles_stack[j].make_collision(particles_stack[j])
                particles_stack[i].mass = 0
                particles_stack[i].not_empty = False
                    # break
        if count % 100000 == 0:
            if count % 1000000 == 0:
                print('t = ', dtau * count)
            fraction.append(np.sum([1 for i in range(particles_stack.size) if particles_stack[i].not_empty == True]) / N)
            time.append(dtau * count)
        count += 1
    plt.plot(time, fraction)
    plt.show()
    print(fraction)
