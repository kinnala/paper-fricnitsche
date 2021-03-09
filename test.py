from skfem import *
from skfem.models.elasticity import linear_elasticity,\
    lame_parameters, linear_stress
from skfem.models.helpers import dot, ddot,\
    prod, sym_grad
import numpy as np
from skfem.io import from_meshio
from skfem.io.json import from_file, to_file
from pathlib import Path


m = MeshTri.init_sqsymmetric().refined(4).translated((0., -.5))

M = (
    (MeshLine(np.linspace(0, 1, 6)) * MeshLine(np.linspace(-1, 1, 10)))
    .translated((1.0, 0.0))
    .refined(3)
    .to_meshtri()
)

M = MeshTri.init_sqsymmetric().refined(4).translated((1., -.5))

#m = m.refined()
M = M.refined()

# define elements and bases
e1 = ElementTriP1()
e = ElementVector(e1)

#E1 = ElementQuad1()
#E = ElementVector(E1)
E1 = ElementTriP1()
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
nu = 0.45
Lambda, Mu = lame_parameters(E, nu)

weakform1 = linear_elasticity(Lambda, Mu)
weakform2 = linear_elasticity(Lambda, Mu)
C = linear_stress(Lambda, Mu)

alpha = 0.0001
limit = 0.5
limit2 = 0.0

def indicator(y):
    return (np.abs(y) >= limit2)

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
        def bilin_mortar(u, v, w):
            ju = (-1.) ** j * dot(u, w.n)
            jv = (-1.) ** i * dot(v, w.n)
            t = w.n.copy()
            t[0] = w.n[1]
            t[1] = -w.n[0]
            nxn = prod(w.n, w.n)
            nxt = prod(w.n, t)
            mu = .5 * ddot(nxn, C(sym_grad(u)))
            mv = .5 * ddot(nxn, C(sym_grad(v)))

            jut = (-1.) ** j * dot(u, t)
            jvt = (-1.) ** i * dot(v, t)
            mut = .5 * ddot(nxt, C(sym_grad(u)))
            mvt = .5 * ddot(nxt, C(sym_grad(v)))

            if j==1:
                mu = 0. * mu
                mut = 0. * mut
            else:
                mu = 2. * mu
                mut = 2. * mut

            if i==1:
                mv = 0. * mv
                mvt = 0. * mvt
            else:
                mv = 2. * mv
                mvt = 2. * mvt


            normal = ((1. / (alpha * w.h) * ju * jv - mu * jv - mv * ju + alpha * w.h * mu * mv)
                      * (np.abs(w.x[1]) <= limit))
            tangent = ((1. / (alpha * w.h) * jut * jvt - mut * jvt - mvt * jut + alpha * w.h * mut * mvt)
                        * (indicator(w.x[1])))
            #tangent2 = alpha * w.h * mut * mvt * (np.abs(w.x[1]) > limit2)

            return normal + tangent #+ tangent2

        K[i][j] += asm(bilin_mortar, mb[j], mb[i])

    @LinearForm
    def lin_mortar(v, w):
        t = w.n.copy()
        t[0] = w.n[1]
        t[1] = -w.n[0]
        jv = (-1.) ** i * dot(v, t)
        #jv = dot(v, t)
        s = 3e-5
        ind = 1 - (indicator(w.x[1]))
        return s * ind * jv
        #mv = .5 * ddot(prod(w.n, w.n), C(sym_grad(v)))
        #return ((1. / (alpha * w.h) * gap(w.x) * jv - gap(w.x) * mv)
        #        * (np.abs(w.x[1]) <= limit))

    f[i] = 0*asm(lin_mortar, mb[i])

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
#x[ib.get_dofs(lambda x: x[0] == 0.0).facet['u^1']] = 0.1

x = solve(*condense(K, f, I=I, x=x))


#e_dg = ElementTriDG(ElementTriP1())
#E_dg = ElementQuadDG(ElementQuad1())
e_dg = ElementTriDG(ElementTriP0())
E_dg = ElementTriDG(ElementTriP0())
#E_dg = ElementQuadDG(ElementQuad0())

fbasis = ExteriorFacetBasis(m, e_dg, facets=m.facets_satisfying(lambda x: x[0] == 1))

# create a displaced mesh
sf = 1

m.p[0] = m.p[0] + sf * x[i1][ib.nodal_dofs[0]]
m.p[1] = m.p[1] + sf * x[i1][ib.nodal_dofs[1]]

M.p[0] = M.p[0] + sf * x[i2][Ib.nodal_dofs[0]]
M.p[1] = M.p[1] + sf * x[i2][Ib.nodal_dofs[1]]


# post processing
s, S = {}, {}



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

ibasis1, y1 = fbasis.trace(s[0, 0], lambda p: p[1], ElementTriP1())
ibasis2, y2 = fbasis.trace(s[0, 1], lambda p: p[1], ElementTriP1())

mini = np.max([np.min(vonmises1), np.min(vonmises2)])
maxi = np.min([np.max(vonmises1), np.max(vonmises2)])
vonmises1 = np.clip(vonmises1, 0.0, 0.1)
vonmises2 = np.clip(vonmises2, 0.0, 0.1)
#vonmises2 = np.clip(vonmises2, np.min(vonmises1), np.max(vonmises1))

from os.path import splitext
from sys import argv
from skfem.visuals.matplotlib import *
ax = plot(ib_dg, vonmises1, Nrefs=3, shading='gouraud')
#ax = plot(ib_dg, s[0,0], Nrefs=3, shading='gouraud')
draw(m, ax=ax)
plot(Ib_dg, vonmises2, ax=ax, Nrefs=3, shading='gouraud')
#plot(Ib_dg, S[0,0], ax=ax, Nrefs=3, shading='gouraud')
draw(M, ax=ax)
show()
plot(ibasis1, y1, Nrefs=0)
show()
plot(ibasis2, y2, Nrefs=0)
show()
