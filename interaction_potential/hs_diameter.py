"""Module for effective hard-sphere diameters"""
import numpy as np
import scipy.optimize as SciOpt
import scipy.integrate as SciInt

def calc_dhs_bh(pot, T):
    '''Calculate Barker-Henderson hard-sphere diameter [m].'''
    if pot.hardcore:
        return pot.sigma
    sigma = pot.sigmaeff
    def integrand(w):
        Phi = pot.calc_pot_divk(sigma*w)
        return (1-np.exp(-Phi/T))

    # Integration using the substitution w = r/sigma -> dr = sigma*dw
    integral, error = SciInt.quad(func=integrand, a=0, b=1.0,
                                    epsabs=sigma*1e-10, limit=10000)
    assert error/integral<1e-4, error
    dhs = integral*sigma
    return dhs

def calc_dhs_rep(pot, T):
    '''A hard-sphere diameter calculated from the repulsive region, Eq. 4 in Noro-Frenkel paper (m)'''
    if pot.hardcore:
        return pot.sigma
    sigma = pot.sigmaeff
    epsdivk = pot.epsdivkeff
    def integrand(w):
        Phi = pot.calc_pot_divk(sigma*w) + epsdivk
        return (1-np.exp(-Phi/T))

    # Integration using the substitution w = r/sigma -> dr = sigma*dw
    integral, error = SciInt.quad(func=integrand, a=0, b=pot.rmin/sigma,
                                    epsabs=sigma*1e-10, limit=10000)
    assert error/integral<1e-4, error
    dhs = integral*sigma
    return dhs

def calc_dhs_wca(pot, T, rho):
    '''Calculate Weeks-Chandler-Andersen hard-sphere diameter [m].'''
    if pot.hardcore:
        return pot.sigma
    sigma = pot.sigmaeff
    epsdivk = pot.epsdivkeff
    rmin = pot.rmin
    def resid(d):
        if d>rmin:
            return np.inf
        if d<=0:
            return -np.inf
        def integrand_left(r):
            Phi = np.exp(-(pot.calc_pot_divk(r)+epsdivk)/T)
            return Phi*y_hs_desousa_amotz(r, rho, d)

        def integrand_right(r):
            Phi_m1 = np.exp(-(pot.calc_pot_divk(r)+epsdivk)/T) - 1
            return Phi_m1*y_hs_desousa_amotz(r, rho, d)

        integral_left, error = SciInt.quad(func=integrand_left, a=1e-10*sigma, b=d,
                                            epsabs=sigma*1e-10, limit=10000)
        integral_right, error = SciInt.quad(func=integrand_right, a=d, b=rmin,
                                            epsabs=sigma*1e-10, limit=10000)

        return integral_left + integral_right
    sol = SciOpt.root(resid, x0=sigma)
    return sol.x[0]

def calc_dhs_wca_lado(pot, T, rho):
    '''Calculate Weeks-Chandler-Andersen hard-sphere diameter following Lado [m].'''
    if pot.hardcore:
        return pot.sigma
    sigma = pot.sigmaeff
    epsdivk = pot.epsdivkeff
    rmin = pot.rmin
    def resid(d):
        if d<0.1*sigma:
            return np.inf

        def integrand_left(r):
            Phi = np.exp(-(pot.calc_pot_divk(r)+epsdivk)/T)
            return np.exp(-(pot.calc_pot_divk(r)+epsdivk)/T)*dy_hs_desousa_amotz_ddhs(r, rho, d)

        def integrand_right(r):
            Phi_m1 = np.exp(-(pot.calc_pot_divk(r)+epsdivk)/T) - 1
            return Phi_m1*dy_hs_desousa_amotz_ddhs(r, rho, d)

        integral_left, error = SciInt.quad(func=integrand_left, a=1e-10*sigma, b=d,
                                            epsabs=sigma*1e-10, limit=10000)
        integral_right, error = SciInt.quad(func=integrand_right, a=d, b=rmin,
                                            epsabs=sigma*1e-10, limit=10000)

        return integral_left + integral_right
    sol = SciOpt.root(resid, x0=sigma)
    return sol.x[0]


def calc_dhs_BFC(pot, T):
    '''Calculate hard-sphere diameter according to Boltzmann-Factor-Criterion [m].'''
    if pot.hardcore:
        return pot.sigma
    def resid(r):
        return pot.calc_pot_divk(r)/T-1
    sol = SciOpt.root(resid, x0=pot.sigma)
    return sol.x[0]


def y_hs_desousa_amotz(r, rho, dhs):
    '''Analytic approximation of hard sphere cavity correlation function y
    at position r, valid for pure hard-spheres. The approximation is
    due to de Souza and Ben-Amotz (10.1080/00268979300100131)

    '''
    # if r>1.5*dhs:
    #     return 1.0
    eta = np.pi/6 * rho * dhs**3
    denom = (1.0-eta)**3
    A = (3.0-eta)/denom - 3.0
    B = -3*eta*(2.0-eta)/denom
    C = np.log((2.0-eta)*(2*denom)) - eta*(2.0-6*eta + 3*eta**2)/denom
    y = np.exp(A + B*(r/dhs) + C*(r/dhs)**3)
    return y

def dy_hs_desousa_amotz_ddhs(r, rho, dhs):
    delta = dhs*1e-5
    numdiff = (y_hs_desousa_amotz(r, rho, dhs+delta) - y_hs_desousa_amotz(r, rho, dhs-delta))/(2*delta)
    return numdiff
