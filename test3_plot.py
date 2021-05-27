import matplotlib.pyplot as plt
from numpy import genfromtxt

uniform = genfromtxt('uniform.csv', delimiter=',')
adaptive = genfromtxt('adaptive.csv', delimiter=',')

plt.loglog(uniform[:, 1], uniform[:, 3], 'x-')
plt.loglog(adaptive[:-1, 0], adaptive[:-1, 2], 'x:')
plt.xlabel('$N$')
plt.ylabel('$\eta$')
plt.grid('major')
plt.legend(['Uniform', 'Adaptive'])
plt.savefig('test3_plot_convergence.pdf')
