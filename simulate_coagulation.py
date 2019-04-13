#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.integrate import odeint

import copy
import time

from dataclasses import dataclass, field
from typing import List

@dataclass
class Profiler(object):
    message : str = field(default = '')
    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print ("{}\n\tElapsed time: {:.3f} sec".format(self.message, time.time() - self._startTime))

@dataclass
class Particle:
    mass : float
    not_empty : bool = True
    velocity : float = field(init = False)

    def make_collision(self, other):
        self.mass += other.mass

@dataclass
class Cluster(Particle):
    particles : List[Particle] = field(default_factory = list)
    number_particles : int = field(init = False)

@dataclass
class Model:
    time : float
    coagulation_matrix : np.ndarray
    particles_stack : List[Cluster] = field(default_factory = list, init = True)
    particles_num : int = 0
    # _t : List[float] = field(default_factory = list)

    def __post_init__(self):
        self.particles_number = len(self.particles_stack)
        self._tic = self.__calculate_step_time(self.coagulation_matrix)

    def __calculate_step_time(self, Betta) -> float:
        return 1 / np.sum([[Betta[i, j] for j in range(Betta.shape[1]) if i != j] for i in range(Betta.shape[0])])

    def __choice_coagulation_pair_FDSMC(self, Betta, i, j) -> bool:
        # random.seed(10)
        R = random.random()
        beg, mid, top = 0,  np.sum([[Betta[k, l] for l in range(Betta.shape[1]) if k != l] for k in range(Betta.shape[0])]), 0
        for k in range(i):
            for l in range(j + 1):
                if k != l:
                    if l < j:
                        beg += Betta[k, l]
                    top += Betta[k, l]
        return beg <= R * mid and R * mid <= top

    def __choice_coagulation_pair(self, Betta, i, j) -> bool:
        return random.choice([0, 1]) < (Betta[i, j] / np.max(Betta))

    def __choice_coagulation_pair_FDSMC(self, Betta, i, j) -> bool:
        random.seed()
        R = random.random()
        beg, mid, top = 0,  np.sum([[Betta[k, l] for l in range(Betta.shape[1]) if k != l] for k in range(Betta.shape[0])]), 0
        for k in range(i):
            for l in range(j + 1):
                if k != l:
                    if l < j:
                        beg += Betta[k, l]
                    top += Betta[k, l]
        return beg <= R * mid and R * mid <= top

    def modeling(self):
        particles_num = len(particles_stack)
        count = 1
        fraction = []
        time_list = []
        while self._tic * count <= self.time:
            # Choice pair coagulation
            i = random.choice([i for i in range(len(self.particles_stack) - 1)])
            j = random.choice([j for j in range(i, len(self.particles_stack))])
            pr = self.__choice_coagulation_pair_FDSMC(self.coagulation_matrix, i, j)
            # If aggregation occurs
            if pr:
                self.coagulation_matrix = np.delete(self.coagulation_matrix, i, 0)
                self.coagulation_matrix = np.delete(self.coagulation_matrix, i, 1)
                self._tic = self.__calculate_step_time(self.coagulation_matrix)
                self.particles_stack[j].make_collision(self.particles_stack[j])
                self.particles_stack = np.hstack((self.particles_stack[:i-1], self.particles_stack[i:]))

            if count % (len(self.particles_stack) * 2) == 0 or self._tic * count >= self.time:
                print('t = {}, dtau = {}'.format(self._tic * count, self._tic))
                fraction.append(np.sum([1 for i in range(self.particles_stack.size) if self.particles_stack[i].mass != 0]) / particles_num)
                time_list.append(self._tic * count)
            count += 1
        return fraction, time_list

def dndt(n, t, J, C, N):
    dndt = np.empty(N)
    for k in range(N):
        dndt[k] = 0.5 * np.sum([C[k-i, i] * n[i]  * n[k-i] for i in range(k - 1)]) - n[k] * np.sum([C[k, i] * n[i] for i in range(N)])
    return dndt

if __name__ == '__main__':

    N = 200
    Ph = np.empty((N, N))
    with Profiler(message='Create Ph'):
        Ph[:, :] = [[((i + 1) ** (0.75) * (j + 1) ** (-0.75) + (i + 1) ** (-0.75) * (j + 1) ** (0.75)) for j in range(N)] for i in range(N)]
    t = 1
    with Profiler(message='Create particles stack'):
        particles_stack = np.array([Particle(mass=1) for i in range(N)])

    model = Model(time=1.0, coagulation_matrix=Ph, particles_stack=particles_stack)
    fraction, time_list = model.modeling()
    print(len(fraction))
    
    dt = 0.1
    t = np.arange(0, 1 + dt, dt)
    tolerance = 1e-6
    J = 1.
    C = np.empty((N, N))
    C[:, :] = [[((i + 1) ** (0.75) * (j + 1) ** (-0.75) + (i + 1) ** (-0.75) * (j + 1) ** (0.75)) for j in range(N)] for i in range(N)]
    # C = copy.deepcopy(Ph)
    n = np.zeros(N)
    n[1] = 1
    with Profiler('ODE') as sec:
        dn = odeint(dndt, n, t, (J, C, N))
    dN = np.empty(dn.shape[0])
    q = 0
    for i in range(dn.shape[0]):
        dN[i] = i ** q * np.sum(dn[i, :])
    plt.plot(t, dN)
    plt.scatter(time_list, fraction, marker='+', c='c')

    plt.show()
