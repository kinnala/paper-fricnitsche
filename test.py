from skfem import *
from skfem.models.elasticity import linear_elasticity,\
    lame_parameters, linear_stress
from skfem.models.helpers import dot, ddot,\
    prod, sym_grad
import numpy as np
from skfem.io import from_meshio
from skfem.io.json import from_file, to_file
from pathlib import Path


m = MeshTri.init_sqsymmetric().refined(3).translated((0., -.5))

M = (
    (MeshLine(np.linspace(0, 1, 6)) * MeshLine(np.linspace(-1, 1, 10)))
    .translated((1.0, 0.0))
    .refined(2)
)

# define elements and bases
e1 = ElementTriP2()
e = ElementVector(e1)

E1 = ElementQuad2()
E = ElementVector(E1)

ib = InteriorBasis(m, e, intorder=4)
Ib = InteriorBasis(M, E, intorder=4)

mapping = MappingMortar.init_2D(
    m,
    M,
    m.facets_satisfying(lambda x: x[0] == 1.0),
    M.facets_satisfying(lambda x: x[0] == 1.0),
    np.array([0.0, 1.0])
)

mb = [
    MortarFacetBasis(m, e, mapping=mapping, intorder=4, side=0),
    MortarFacetBasis(M, E, mapping=mapping, intorder=4, side=1),
]

# define bilinear forms
E = 1.0
nu = 0.3
Lambda, Mu = lame_parameters(E, nu)

weakform1 = linear_elasticity(Lambda, Mu)
weakform2 = linear_elasticity(Lambda, Mu)
C = linear_stress(Lambda, Mu)

alpha = 0.01
limit = 0.5

# assemble the stiffness matrices
K1 = asm(weakform1, ib)
K2 = asm(weakform2, Ib)
K = [[K1, 0.], [0., K2]]
f = [None] * 2


def gap(x):
    """Initial gap between the bodies."""
    return 0.0 * x[0]

# assemble the mortar matrices
for i in range(2):
    for j in range(2):

        @BilinearForm
        def bilin_mortar_orig(u, v, w):
            ju = (-1.) ** j * dot(u, w.n)
            jv = (-1.) ** i * dot(v, w.n)
            nxn = prod(w.n, w.n)
            mu = .5 * ddot(nxn, C(sym_grad(u))) * (i == 0)
            mv = .5 * ddot(nxn, C(sym_grad(v))) * (i == 0)
            return ((1. / (alpha * w.h) * ju * jv - mu * jv - mv * ju)
                    * (np.abs(w.x[1]) <= limit))

        @BilinearForm
        def bilin_mortar(u, v, w):
            ju = (-1.) ** j * u
            jv = (-1.) ** i * v
            nxn = prod(w.n, w.n)
            sigma = .5 * C(sym_grad(u)) * (i == 0)
            tau = .5 * C(sym_grad(v)) * (i == 0)
            # mu = .5 * ddot(nxn, C(sym_grad(u))) * (i == 0)
            # mv = .5 * ddot(nxn, C(sym_grad(v))) * (i == 0)
            return ((1. / (alpha * w.h) * dot(ju, jv)
                     - ddot(sigma, prod(jv, w.n))
                     - ddot(tau, prod(ju, w.n)))
                    * (np.abs(w.x[1]) <= limit))

        K[i][j] += asm(bilin_mortar, mb[j], mb[i])

    @LinearForm
    def lin_mortar(v, w):
        jv = (-1.) ** i * dot(v, w.n)
        mv = .5 * ddot(prod(w.n, w.n), C(sym_grad(v)))
        return ((1. / (alpha * w.h) * gap(w.x) * jv - gap(w.x) * mv)
                * (np.abs(w.x[1]) <= limit))

    f[i] = asm(lin_mortar, mb[i])

import scipy.sparse
K = (scipy.sparse.bmat(K)).tocsr()

# set boundary conditions and solve
i1 = np.arange(K1.shape[0])
i2 = np.arange(K2.shape[0]) + K1.shape[0]

D1 = ib.get_dofs(lambda x: x[0] == 0.0).all()
D2 = Ib.get_dofs(lambda x: x[0] == 2.0).all()

x = np.zeros(K.shape[0])

f = np.hstack((f[0], f[1]))

x = np.zeros(K.shape[0])
D = np.concatenate((D1, D2 + ib.N))
I = np.setdiff1d(np.arange(K.shape[0]), D)

x[ib.get_dofs(lambda x: x[0] == 0.0).nodal['u^1']] = 0.1
x[ib.get_dofs(lambda x: x[0] == 0.0).facet['u^1']] = 0.1

x = solve(*condense(K, f, I=I, x=x))


# create a displaced mesh
sf = 1

m.p[0] = m.p[0] + sf * x[i1][ib.nodal_dofs[0]]
m.p[1] = m.p[1] + sf * x[i1][ib.nodal_dofs[1]]

M.p[0] = M.p[0] + sf * x[i2][Ib.nodal_dofs[0]]
M.p[1] = M.p[1] + sf * x[i2][Ib.nodal_dofs[1]]


# post processing
s, S = {}, {}
e_dg = ElementTriDG(ElementTriP1())
E_dg = ElementQuadDG(ElementQuad1())

for itr in range(2):
    for jtr in range(2):

        @BilinearForm
        def proj_cauchy(u, v, w):
            return C(sym_grad(u))[itr, jtr] * v

        @BilinearForm
        def mass(u, v, w):
            return u * v

        ib_dg = InteriorBasis(m, e_dg, intorder=4)
        Ib_dg = InteriorBasis(M, E_dg, intorder=4)

        s[itr, jtr] = solve(asm(mass, ib_dg),
                            asm(proj_cauchy, ib, ib_dg) @ x[i1])
        S[itr, jtr] = solve(asm(mass, Ib_dg),
                            asm(proj_cauchy, Ib, Ib_dg) @ x[i2])

s[2, 2] = nu * (s[0, 0] + s[1, 1])
S[2, 2] = nu * (S[0, 0] + S[1, 1])

vonmises1 = np.sqrt(.5 * ((s[0, 0] - s[1, 1]) ** 2 +
                          (s[1, 1] - s[2, 2]) ** 2 +
                          (s[2, 2] - s[0, 0]) ** 2 +
                          6. * s[0, 1]**2))

vonmises2 = np.sqrt(.5 * ((S[0, 0] - S[1, 1]) ** 2 +
                          (S[1, 1] - S[2, 2]) ** 2 +
                          (S[2, 2] - S[0, 0]) ** 2 +
                          6. * S[0, 1]**2))



from os.path import splitext
from sys import argv
from skfem.visuals.matplotlib import *
ax = plot(ib_dg, vonmises1, shading='gouraud')
draw(m, ax=ax)
plot(Ib_dg, vonmises2, ax=ax, Nrefs=3, shading='gouraud')
draw(M, ax=ax)
show()
