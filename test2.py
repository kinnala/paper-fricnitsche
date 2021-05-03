from skfem import *
from skfem.models.elasticity import (linear_elasticity,
                                     lame_parameters,
                                     linear_stress)
from skfem.models.helpers import dot, ddot, prod, sym_grad
import numpy as np

m = MeshTri.init_sqsymmetric().refined(4).translated((0., -.5))

e1 = ElementTriP1()
e = ElementVector(e1)

basis = InteriorBasis(m, e, intorder=4)

fbasis = FacetBasis(m, e, intorder=4,
                    facets=m.facets_satisfying(lambda x: x[0] == 1))

E = 1.0
nu = 0.3
Lambda, Mu = lame_parameters(E, nu)

weakform = linear_elasticity(Lambda, Mu)
C = linear_stress(Lambda, Mu)

alpha = 1e-2
kappa = 0.03

K = asm(weakform, basis)


def gap(x):
    """Initial gap between the bodies."""
    return 0.0 * x[0]

@BilinearForm
def nitsche(u, v, w):

    uprev = w['prev']
    x, y = w.x

    # helper vectors
    t = w.n.copy()
    t[0] = w.n[1]
    t[1] = -w.n[0]
    nxn = prod(w.n, w.n)
    nxt = prod(w.n, t)

    # components
    ut = dot(u, t)
    un = dot(u, w.n)
    vt = dot(v, t)
    vn = dot(v, w.n)

    # sigma(u)n = sun, sigma(v)n = svn
    sun = ddot(nxn, C(sym_grad(u)))
    svn = ddot(nxn, C(sym_grad(v)))
    sut = ddot(nxt, C(sym_grad(u)))
    svt = ddot(nxt, C(sym_grad(v)))

    lambdat = 1. / (alpha * w.h) * dot(uprev, t) - ddot(nxt, C(sym_grad(uprev)))

    normal = (1. / (alpha * w.h) * un * vn - sun * vn - svn * un + alpha * w.h * sun * svn)
    tangent = (1. / (alpha * w.h) * ut * vt - sut * vt - svt * ut + alpha * w.h * sut * svt) * (np.abs(lambdat) < kappa)

    return normal + tangent

@LinearForm
def nitsche_load(v, w):

    uprev = w['prev']
    x, y = w.x

    # helper vectors
    t = w.n.copy()
    t[0] = w.n[1]
    t[1] = -w.n[0]
    nxn = prod(w.n, w.n)
    nxt = prod(w.n, t)

    # components
    vt = dot(v, t)
    vn = dot(v, w.n)
    svn = ddot(nxn, C(sym_grad(v)))
    svt = ddot(nxt, C(sym_grad(v)))

    skappa = kappa * np.sign(y)

    lambdat = 1. / (alpha * w.h) * dot(uprev, t) - ddot(nxt, C(sym_grad(uprev)))

    return skappa * vt * (np.abs(lambdat) >= kappa)

xprev = basis.zeros()

for itr in range(10):

    print(itr)

    B = asm(nitsche, fbasis, prev=fbasis.interpolate(xprev))

    f = asm(nitsche_load, fbasis, prev=fbasis.interpolate(xprev))

    D = basis.get_dofs(lambda x: x[0] == 0.0)

    x = np.zeros(K.shape[0])
    x[D.nodal['u^1']] = 0.1

    x = solve(*condense(K + B, f, D=D, x=x))

    print(np.linalg.norm(x - xprev))
    xprev = x.copy()


# calculate stress
e_dg = ElementTriDG(ElementTriP0())
basis_dg = InteriorBasis(m, e_dg, intorder=4)
fbasis_dg = FacetBasis(m, e_dg, facets=m.facets_satisfying(lambda x: x[0] == 1.0))
s = {}

for itr in range(2):
    for jtr in range(2):

        @BilinearForm
        def proj_cauchy(u, v, w):
            return C(sym_grad(u))[itr, jtr] * v

        @BilinearForm
        def mass(u, v, w):
            return u * v

        s[itr, jtr] = solve(asm(mass, basis_dg),
                            asm(proj_cauchy, basis, basis_dg) @ x)

# off-plane component
s[2, 2] = nu * (s[0, 0] + s[1, 1])

vonmises = np.sqrt(.5 * ((s[0, 0] - s[1, 1]) ** 2 +
                         (s[1, 1] - s[2, 2]) ** 2 +
                         (s[2, 2] - s[0, 0]) ** 2 +
                         6. * s[0, 1]**2))

tbasis_n, Lam_n = fbasis_dg.trace(s[0, 0], lambda p: p[1], ElementTriP0())
tbasis_t, Lam_t = fbasis_dg.trace(s[0, 1], lambda p: p[1], ElementTriP0())


# plotting
from skfem.visuals.matplotlib import *

# stresses
ax = plot(basis_dg, vonmises, Nrefs=3, shading='gouraud')
mdefo = m.translated(x[basis.nodal_dofs])
draw(mdefo, ax=ax)

# normal lagmult
plot(tbasis_n, Lam_n, Nrefs=0, color='k.')

# tangential lagmult
plot(tbasis_t, Lam_t, Nrefs=0, color='k.')

show()
