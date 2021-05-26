from skfem import *
from skfem.models.elasticity import (linear_elasticity,
                                     lame_parameters,
                                     linear_stress)
from skfem.helpers import dot, ddot, prod, sym_grad, grad
import numpy as np

m = (MeshTri
     .init_sqsymmetric()
     .refined(1)
     .translated((0., -.5)))

maxiters = 50

for k in range(maxiters):

    e1 = ElementTriP2()
    e = ElementVector(e1)

    basis = InteriorBasis(m, e, intorder=4)

    fbasis = FacetBasis(m, e, intorder=4,
                        facets=m.facets_satisfying(lambda x: x[0] == 1))

    E = 1.0
    nu = 0.3
    Lambda, Mu = lame_parameters(E, nu)

    weakform = linear_elasticity(Lambda, Mu)
    C = linear_stress(Lambda, Mu)

    alpha = 1e-3
    kappa = 0.02
    alternative = False

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


        normal = (1. / (alpha * w.h) * un * vn - sun * vn - svn * un + alpha * w.h * sun * svn)
        if alternative:
            lambdat = 1. / (alpha) * dot(uprev, t) - ddot(nxt, C(sym_grad(uprev)))
            tangent = (1. / (alpha) * ut * vt - sut * vt - svt * ut + alpha * sut * svt) * (np.abs(lambdat) < kappa)
        else:
            lambdat = 1. / (alpha * w.h) * dot(uprev, t) - ddot(nxt, C(sym_grad(uprev)))
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

        if alternative:
            lambdat = 1. / (alpha) * dot(uprev, t) - ddot(nxt, C(sym_grad(uprev)))
        else:
            lambdat = 1. / (alpha * w.h) * dot(uprev, t) - ddot(nxt, C(sym_grad(uprev)))

        return skappa * vt * (np.abs(lambdat) >= kappa)

    xprev = basis.zeros()

    for itr in range(40):

        B = asm(nitsche, fbasis, prev=fbasis.interpolate(xprev))

        f = asm(nitsche_load, fbasis, prev=fbasis.interpolate(xprev))

        D = basis.get_dofs(lambda x: x[0] == 0.0)

        x = np.zeros(K.shape[0])
        x[D.nodal['u^1']] = 0.1
        x[D.facet['u^1']] = 0.1

        x = solve(*condense(K + B, f, D=D.all('u^1'), x=x))

        diff = np.linalg.norm(x - xprev)

        if diff < 1e-8:
            break
        xprev = x.copy()


    # calculate stress
    e_dg = ElementTriDG(ElementTriP1())
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

    tbasis_n, Lam_n = fbasis_dg.trace(-s[0, 0], lambda p: p[1], ElementTriP1())
    tbasis_t, Lam_t = fbasis_dg.trace(s[0, 1], lambda p: p[1], ElementTriP1())

    # plotting
    from skfem.visuals.matplotlib import *
    import matplotlib.pyplot as plt

    # calculate error estimators

    # interior residual

    @Functional
    def divsigma(w):
        return w.h ** 2 * (w['s0'].grad[0] + w['s1'].grad[1]) ** 2

    est1 = divsigma.elemental(basis_dg,
                              s0=basis_dg.interpolate(s[0, 0]),
                              s1=basis_dg.interpolate(s[0, 1]))
    est2 = divsigma.elemental(basis_dg,
                              s0=basis_dg.interpolate(s[1, 0]),
                              s1=basis_dg.interpolate(s[1, 1]))

    eta_K = est1 + est2

    ## interior edge jump estimator
    def edge_estimator(m, s, ix):

        fbasis = [
            InteriorFacetBasis(m, e_dg, intorder=4, side=0),
            InteriorFacetBasis(m, e_dg, intorder=4, side=1),
        ]
        ws = {
            'plus0': fbasis[0].interpolate(s[ix, 0]),
            'plus1': fbasis[0].interpolate(s[ix, 1]),
            'minus0': fbasis[1].interpolate(s[ix, 0]),
            'minus1': fbasis[1].interpolate(s[ix, 1]),
        }

        @Functional
        def edge_jump(w):
            h = w.h
            n = w.n
            return h * ((w['plus0'] - w['minus0']) * n[0]
                        + (w['plus1'] - w['minus1']) * n[1]) ** 2

        eta_E1 = edge_jump.elemental(fbasis[0], **ws)

        tmp = np.zeros(m.facets.shape[1])
        np.add.at(tmp, fbasis[0].find, eta_E1)
        eta_E1 = np.sum(.5 * tmp[m.t2f], axis=0)

        return eta_E1

    eta_E  = edge_estimator(m, s, 0) + edge_estimator(m, s, 1)

    est = eta_K + eta_E

    err = np.sqrt(Functional(lambda w: w['sol'].value[0] ** 2 + w['sol'].value[1] ** 2 + ddot(grad(w['sol']), grad(w['sol'])))
                  .assemble(basis, sol=basis.interpolate(x)))
    print("{},{},{}".format(len(x), err, np.sqrt(np.sum(est))))

    # plots
    if k == maxiters - 1 or len(x) > 7900:
        # estimators
        #plot(m, est1)

        # stresses
        ax = plot(basis_dg, s[0, 1], Nrefs=3, shading='gouraud', colorbar=True)

        mdefo = m.translated(x[basis.nodal_dofs])
        draw(mdefo)

        # normal lagmult
        plot(tbasis_n, Lam_n, Nrefs=1, color='k.')

        # tangential lagmult
        plot(tbasis_t, Lam_t, Nrefs=1, color='k.')

        break

    m = m.refined(adaptive_theta(est, theta=0.05))

show()

