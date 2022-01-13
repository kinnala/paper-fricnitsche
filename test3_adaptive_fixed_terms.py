from skfem import *
from skfem.models.elasticity import (linear_elasticity,
                                     lame_parameters,
                                     linear_stress)
from skfem.helpers import dot, ddot, prod, sym_grad, grad
import numpy as np
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

m = (MeshTri
     .init_sqsymmetric()
     .refined(1)
     .translated((0., -.5)))

maxiters = 50

kappa = 0.02

def indicator(lambdat):
    return np.abs(lambdat).mean(axis=1) < kappa

alldiffs = {}

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
    alternative = False

    K = asm(weakform, basis)


    def gap(x):
        """Initial gap between the bodies."""
        return -0.1 + 0.0 * x[0]

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

        normal = (1. / (alpha * w.h) * un * vn - sun * vn - svn * un)

        lambdat = 1. / (alpha * w.h) * dot(uprev, t) - ddot(nxt, C(sym_grad(uprev)))
        ind = indicator(lambdat)

        tangent = (1. / (alpha * w.h) * ut * vt - sut * vt - svt * ut) * ind[:, None] - alpha * w.h * sut * svt * (~ind[:, None])

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

        ind = ~indicator(lambdat)

        return skappa * vt * ind[:, None] - skappa * w.h * alpha * svn * ind[:, None] + (1. / (alpha * w.h) * gap(w.x) * vn - svn * gap(w.x))

    xprev = basis.zeros()
    diffs = []
    for itr in range(40):

        B = asm(nitsche, fbasis, prev=fbasis.interpolate(xprev))

        g = asm(nitsche_load, fbasis, prev=fbasis.interpolate(xprev))

        D = basis.get_dofs(lambda x: x[0] == 0.0)

        x = np.zeros(K.shape[0])

        x = solve(*condense(K + B, g, D=D.all(), x=x))

        diff = np.sqrt(K.dot(x - xprev).dot(x - xprev))
        diffs.append(diff)
        if diff < 1e-10:
            break
        xprev = x.copy()

    alldiffs[len(xprev)] = diffs


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
            InteriorFacetBasis(m, e_dg, intorder=6, side=0),
            InteriorFacetBasis(m, e_dg, intorder=6, side=1),
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

    ## neumann estimator
    def neumann_estimator(m, s, ind):

        fbasis = FacetBasis(m, e_dg, facets=m.facets_satisfying(ind))
        s1 = [fbasis.interpolate(s[0, i]) for i in [0, 1]]
        s2 = [fbasis.interpolate(s[1, i]) for i in [0, 1]]

        @Functional
        def traction_zero(w):
            h = w.h
            n = w.n
            si1 = w.w1
            si2 = w.w2
            return h * (si1 * n[0] + si2 * n[1]) ** 2

        eta_neumann = (traction_zero.elemental(fbasis, w1=s1[0], w2=s1[1])
                       + traction_zero.elemental(fbasis, w1=s2[0], w2=s2[1]))

        tmp = np.zeros(m.facets.shape[1])
        np.add.at(tmp, fbasis.find, eta_neumann)
        return np.sum(.5 * tmp[m.t2f], axis=0)

    eta_N = neumann_estimator(m, s, lambda x: np.abs(x[1]) == 0.5)

    ## contact estimator
    @Functional
    def contact_estimator(w):
        h = w.h
        n = w.n.copy()
        t = w.n.copy()
        t[0] = w.n[1]
        t[1] = -w.n[0]
        nxn = prod(w.n, w.n)
        nxt = prod(w.n, t)
        lambdan = 1. / (alpha * w.h) * (dot(w['sol'], n) - gap(w.x)) - ddot(nxn, C(sym_grad(w['sol'])))
        gammat = 1. / (alpha * w.h) * dot(w['sol'], t) - ddot(nxt, C(sym_grad(w['sol'])))

        lambdat = gammat * (np.abs(gammat) < kappa) - kappa * np.sign(w.x[1]) * (np.abs(gammat) >= kappa)
        sun = ddot(nxn, C(sym_grad(w['sol'])))
        sut = ddot(nxt, C(sym_grad(w['sol'])))
        return (0. * 1. / h * ((w['sol'].value[0] - gap(w.x)) * (w['sol'].value[0] - gap(w.x) > 0)) ** 2
                + h * (lambdat + sut) ** 2
                + h * (lambdan + sun) ** 2)

    fbasis_G = FacetBasis(m, e, facets=m.facets_satisfying(lambda x: x[0] == 1.))
    eta_G = contact_estimator.elemental(fbasis_G, sol=fbasis_G.interpolate(x))
    tmp = np.zeros(m.facets.shape[1])
    np.add.at(tmp, fbasis_G.find, eta_G)
    eta_G = np.sum(.5 * tmp[m.t2f], axis=0)

    ## S-term
    @Functional
    def S_term(w):
        h = w.h
        n = w.n.copy()
        t = w.n.copy()
        t[0] = w.n[1]
        t[1] = -w.n[0]
        nxn = prod(w.n, w.n)
        nxt = prod(w.n, t)
        lambdan = 1. / (alpha * w.h) * (dot(w['sol'], n) - gap(w.x)) - ddot(nxn, C(sym_grad(w['sol'])))
        gammat = 1. / (alpha * w.h) * dot(w['sol'], t) - ddot(nxt, C(sym_grad(w['sol'])))

        lambdat = gammat * (np.abs(gammat) < kappa) - kappa * np.sign(w.x[1]) * (np.abs(gammat) >= kappa)
        sun = ddot(nxn, C(sym_grad(w['sol'])))
        sut = ddot(nxt, C(sym_grad(w['sol'])))
        # lambdat, sut
        # lambdan, sun
        return ((gap(w.x) - w['sol'].value[0]) * (gap(w.x) - w['sol'].value[0] < 0)) ** 2 + ((gap(w.x) - w['sol'].value[0]) * (gap(w.x) - w['sol'].value[0] > 0)) * lambdan + (kappa * np.abs(w['sol'].value[1]) - np.abs(w['sol'].value[1] * lambdat))

    S_term_val = S_term.elemental(fbasis_G, sol=fbasis_G.interpolate(x))
    tmp = np.zeros(m.facets.shape[1])
    np.add.at(tmp, fbasis_G.find, S_term_val)
    eta_G = np.sum(.5 * tmp[m.t2f], axis=0)

    ## lambda plot
    @Functional
    def lambdat(w):
        h = w.h
        n = w.n.copy()
        t = w.n.copy()
        t[0] = w.n[1]
        t[1] = -w.n[0]
        nxn = prod(w.n, w.n)
        nxt = prod(w.n, t)
        lambdan = 1. / (alpha * w.h) * (dot(w['sol'], n) - gap(w.x)) - ddot(nxn, C(sym_grad(w['sol'])))
        gammat = 1. / (alpha * w.h) * dot(w['sol'], t) - ddot(nxt, C(sym_grad(w['sol'])))
        sun = -ddot(nxn, C(sym_grad(w['sol'])))
        sut = -ddot(nxt, C(sym_grad(w['sol'])))

        lambdat = gammat * (np.abs(gammat) < kappa) - kappa * np.sign(w.x[1]) * (np.abs(gammat) >= kappa)
        import matplotlib.pyplot as plt
        ix = np.argsort(w.x[1].flatten())
        ## NOTE! enable to draw lagmult
        plt.figure(figsize=(4, 3))
        plt.plot(w.x[1].flatten()[ix], lambdan.flatten()[ix], 'k')
        plt.ylabel('$\lambda_n$')
        plt.xlabel('$y$')
        plt.savefig('test3_adaptive_lambdan_{}.pdf'.format(k))
        plt.close()

        plt.figure(figsize=(4, 3))
        plt.plot(w.x[1].flatten()[ix], lambdat.flatten()[ix], 'k')
        plt.ylabel('$\lambda_t$')
        plt.xlabel('$y$')
        plt.savefig('test3_adaptive_lambdat_{}.pdf'.format(k))
        plt.close()

        # draw u_t
        plt.figure(figsize=(4, 3))
        plt.plot(w.x[1].flatten()[ix], dot(w['sol'], t).flatten()[ix], 'k')
        plt.ylabel('$u_t$')
        plt.xlabel('$y$')
        plt.savefig('test3_adaptive_ut_{}.pdf'.format(k))
        plt.close()

        # draw u_n
        plt.figure(figsize=(4, 3))
        plt.plot(w.x[1].flatten()[ix], dot(w['sol'], n).flatten()[ix], 'k')
        plt.ylabel('$u_n$')
        plt.xlabel('$y$')
        plt.ylim([-0.125, -0.075])
        plt.savefig('test3_adaptive_un_{}.pdf'.format(k))
        plt.close()

        return lambdat

    fix = m.facets_satisfying(lambda x: x[0] == 1.)
    fbasis_lambda = FacetBasis(m, e, facets=fix)
    lambdat = lambdat.elemental(fbasis_lambda, sol=fbasis_lambda.interpolate(x))

    ## total estimator
    est = eta_K + eta_E + eta_N + eta_G

    err = np.sqrt(Functional(lambda w: w['sol'].value[0] ** 2 + w['sol'].value[1] ** 2 + ddot(grad(w['sol']), grad(w['sol'])))
                  .assemble(basis, sol=basis.interpolate(x)))
    print("{},{},{},{}".format(len(x), err, np.sqrt(np.sum(est)), np.sqrt(S_term_val.sum())))

    # plots
    if k == maxiters - 1 or len(x) > 7600:
        # estimators
        #plot(m, est1)

        # stresses
        ax = plot(basis_dg, s[0, 1], Nrefs=3, shading='gouraud', colorbar=True)
        plt.savefig('test3_adaptive_s01_{}.pdf'.format(k))
        plt.close()

        mdefo = m.translated(x[basis.nodal_dofs])
        draw(mdefo)
        plt.savefig('test3_adaptive_defo_{}.pdf'.format(k))
        plt.close()

        # displacement
        pbasis = InteriorBasis(m, ElementTriP2())
        ix2 = np.concatenate((basis.nodal_dofs[1], basis.facet_dofs[1]))
        plot(pbasis, x[ix2], Nrefs=2, shading='gouraud', colorbar=True)
        plt.savefig('test3_adaptive_disp_{}.pdf'.format(k))
        plt.close()

        # normal lagmult
        plot(tbasis_n, Lam_n, Nrefs=1, color='k.')
        plt.savefig('test3_adaptive_sigmann_{}.pdf'.format(k))
        plt.close()

        # tangential lagmult
        plot(tbasis_t, Lam_t, Nrefs=1, color='k.')
        plt.savefig('test3_adaptive_sigmant_{}.pdf'.format(k))
        plt.close()

        break

    m = m.refined(adaptive_theta(est, theta=0.7))

import math
NUM_COLORS = math.ceil(21.0/5.0)
cm = plt.get_cmap('gist_rainbow')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
i = 0
alldiffs = {k: alldiffs[k] for i, k in enumerate(alldiffs.keys()) if i % 5 == 0}
for k in alldiffs:
    plt.semilogy(np.array(alldiffs[k]))
legend = list('$N = {}$'.format(k) for k in alldiffs)
plt.xlabel('Contact iteration')
plt.ylabel('Energy norm of the difference')
plt.legend(legend)
plt.savefig('test3_adaptive_contact_convergence.pdf')
plt.close()

