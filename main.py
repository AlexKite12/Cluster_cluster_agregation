#!/usr/bin/env python
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import random

def plot_cube(side_length):
    cube_definition = [
        (0,0,0), (0,side_length,0), (side_length,0,0), (0,0,side_length)
    ]
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]
    return edges, points

def draw_3d_plot(edges, points, clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0))

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)


    for cl in clusters:
        #for prt in cl.coords:
        ax.scatter(cl[0], cl[1], cl[2], s=2, c='b')
    return ax

#calculate number particles in system
def calculate_number_particles(box_length, concentartion):
    return int(concentartion * box_length ** 3)

from dataclasses import dataclass, field
from typing import List

@dataclass
class Particle:
    mass : float
    probability : float
    diffusion : float = None
    coords : List[float] = field(default_factory=np.array)

    @classmethod
    def calculate_diffusion(cls, friction, temperature):
        cls.diffusion = 1.3806503 * 10 ** (-23) / friction * temperature
        return cls.diffusion

@dataclass
class Cluster(Particle):
    number_particles : int = 0

    def calculate_cluster_diffusion(self, diffusity_exponent):
        self.diffusion *= self.mass ** diffusity_exponent
        return self.diffusion

#calculate probability forming new cluster from cluster_one and cluster_two
def calculate_probability_collision(cluster_one, cluster_two,
                                    sticking_probability_exponent, single_probability):
    return single_probability * (cluster_one.number_particles * cluster_two.number_particles) ** sticking_probability_exponent

#calculate probability ptentional movement
def calculate_potential_movement(cluster_one, max_diffusion):
    return cluster_one.diffusion / max_diffusion

#calculate max diffusion in system
def calculate_max_diffusion(clusters_stack):
    return max([cluster.diffusion for cluster in clusters_stack])

def generate_model(side_length, concentartion, probability, temperature, diffusity_exponent):
    particle_mass =  720
    friction = 0.15
    boltzmann_constant = 1.380653 * 10 ** (-23)

    friction_coefficient = 0.2
    mass = 0.1
    probability = 0.1
    diffusion = -0.5

    number_particles = calculate_number_particles(side_length, concentartion)
    clusters_stack = []
    while number_particles > 0:
        number = random.randint(0, number_particles)
        coords = np.random.uniform(0, side_length, 3)
        clusters_stack.append(Cluster(mass, single_probability, coords,
                                Particle.calculate_diffusion(friction, temperature), number))

        clusters_stack[-1].calculate_cluster_diffusion(diffusity_exponent)

        number_particles = number_particles - number
    return clusters_stack

if __name__ == '__main__':

    boltzmann_constant = 1.380653 * 10 ** (-23)

    friction_coefficient = 0.2

    side_length = 80
    concentartion = 0.01
    diffusion_coefficient = -0.5
    probability = 0.5
    single_probability = 0.1
    temperature = 298
    particle_mass =  720
    friction = 0.15
    clusters_stack = generate_model(side_length, concentartion, probability, temperature, diffusion_coefficient)
    print([cluster.number_particles for cluster in clusters_stack])
    clusters_coords = [cluster.coords for cluster in clusters_stack]
    print(clusters_coords)
    draw_3d_plot(*plot_cube(5), clusters_coords)
