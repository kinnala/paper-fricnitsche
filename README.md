# Numerical experiment: Nitsche's method for Tresca contact

This repository contains source code for reproducing the numerical experiment
from the article "Adaptive finite elements for Tresca friction problem" by Tom
Gustafsson and Juha Videman.

## General information

| Author     | Tom Gustafsson                                       |
| University | Aalto University                                     |
| Department | Department of Mathematics and Systems Analysis       |
| Date       | 18.6.2021                                            |
| Funding    | The Academy of Finland (Decisions 324611 and 338341) |

## File overview

### `Makefile`

Example shell commands for running the numerical experiment.

### `test3_fixed_terms.py`

Solves the example problem with uniform mesh refinements.

| License              | The MIT License   |
| Programming language | Python 3.8.3      |
| Dependency           | scipy==1.4.1      |
| Dependency           | numpy==1.20.0     |
| Dependency           | scikit-fem==3.1.0 |
| Dependency           | matplotlib==3.2.2 |

### `test3_adaptive_fixed_terms.py`

Solves the example problem with adaptive mesh refinements.

| License              | The MIT License   |
| Programming language | Python 3.8.3      |
| Dependency           | scipy==1.4.1      |
| Dependency           | numpy==1.20.0     |
| Dependency           | scikit-fem==3.1.0 |
| Dependency           | matplotlib==3.2.2 |

### `test3_plot.py`

Draws a convergence plot using the outputs of `test3_fixed_terms.py` and
`test3_adaptive_fixed_terms.py`.

| License              | The MIT License   |
| Programming language | Python 3.8.3      |
| Dependency           | scipy==1.4.1      |
| Dependency           | numpy==1.20.0     |
| Dependency           | matplotlib==3.2.2 |