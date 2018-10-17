from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def create_cub(concentartion, side_length):
	"""create a cubic box with side-length L and concentartion C"""

	particles_number = round(concentartion * side_length ** 3) #the number of all the single particles N
	particles = side_length * np.random.random_sample((particles_number,3)) #(b - a) * random_sample() + a
	return particles

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    side_length = 10

    #colors = ('r', 'g', 'b', 'k')
    #for c in colors:
    #    x = np.random.sample(5)
    #    y = np.random.sample(5)
    #    ax.scatter(x, y, 0, zdir='y', c=c)
    #x = np.random.sample(5)
    #y = np.random.sample(5)
    particles = create_cub(0.5, side_length)
    for prt in particles:
	    ax.scatter(prt[0], prt[1], prt[2], c='b')

    ax.legend()
    ax.set_xlim3d(0, side_length)
    ax.set_ylim3d(0, side_length)
    ax.set_zlim3d(0, side_length)

    plt.show()