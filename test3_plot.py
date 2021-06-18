import matplotlib.pyplot as plt
from numpy import genfromtxt

uniform = genfromtxt('uniform.csv', delimiter=',')
adaptive = genfromtxt('adaptive.csv', delimiter=',')

plt.loglog(uniform[:, 1], uniform[:, 3], 'x-')
plt.loglog(adaptive[:, 0], adaptive[:, 2], 'o:')
plt.loglog([1e3, 1e4], [1e-2, 10**(-2.5)], 'k-')
plt.loglog([1e3, 1e4], [1e-2, 1e-3], 'k:')
plt.xlabel('$N$')
plt.ylabel('$\eta$')
plt.grid('major')
plt.legend(['Uniform', 'Adaptive', '$O(N^{-1/2})$', '$O(N^{-1})$'])
plt.savefig('test3_plot_convergence.pdf')
