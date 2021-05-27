import matplotlib.pyplot as plt
from numpy import genfromtxt

uniform = genfromtxt('uniform.csv', delimiter=',')
adaptive = genfromtxt('adaptive.csv', delimiter=',')

plt.loglog(uniform[:, 1], uniform[:, 3], 'x-')
plt.loglog(adaptive[:, 0], adaptive[:, 2], 'x-.')
plt.xlabel('$N$')
plt.ylabel('$\eta$')
plt.grid('major')
plt.legend(['Uniform', 'Adaptive'])
plt.show()
