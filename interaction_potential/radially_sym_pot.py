
"""Module implementing radially symmetric potentials."""
from abc import ABC, abstractmethod
import numpy as np
import num_dual as nd
import scipy.integrate as SciInt
import scipy.optimize as SciOpt
import matplotlib.pyplot as plt
from interaction_potential.hs_diameter import *

class RadiallySymmetricPotential(ABC):
    def __init__(self, sigma, epsdivk, precalc=True, hardcore=False, rc=np.inf, shift=False):
        self.sigma = self.sigmaeff = sigma       # (m)
        self.epsdivk = self.epsdivkeff = epsdivk # (K)
        self.hardcore = hardcore # whether the potential is infinite for r<sigma
        self.rc = rc       # truncation position (m)
        self.shift = shift # whether to shift at truncation
        if rc<np.inf:
            self.cut_shift(rc, shift=shift)
        if precalc:
            self.precalc()

    @abstractmethod
    def calc_pot_divk(self, r):
        """Calculate potential
        Args:
            r: separation (m)
        Returns:
            potential value (K)
        """
        pass

    def calc_pot_derivative(self, r):
        """Calculate d/dr pot_divk
        Args:
            r: separation (m)
        Returns:
            dudr: derivative value (K/m)
        """
        (_, dudr) = nd.first_derivative(self.calc_pot_divk, r)
        return dudr

    def cut_shift(self, rc, shift=True):
        full_pot = self.calc_pot_divk
        pot_at_rc = full_pot(rc)
        if abs(pot_at_rc) < np.finfo(float).eps:
            # potential is already zero, no more is needed
            return
        self.shift_correction = pot_at_rc if shift else 0.0

        def calc_pot_divk_cutshift(r):
            """Calculate potential
            """
            if r>self.rc:
                return 0.0
            else:
                return full_pot(r) - self.shift_correction
        self.calc_pot_divk = calc_pot_divk_cutshift
        self.precalc()

    def precalc(self):
        '''Update effective sigma, epsdivk, and rmin.'''
        self.rmin, self.epsdivkeff = self._calc_rmin_and_epseff()
        self.sigmaeff = self._calc_sigmaeff()

    def _calc_rmin_and_epseff(self):
        """Calculate effective epsilon and location of minimum
        Args:
            T (K, optional)
        Returns:
            rmin (m)
            sigmaeff (m)
        """
        def fun(x):
            return self.calc_pot_divk(x*self.sigma)
        res = SciOpt.minimize(fun, x0=1.1, tol=1e-7, bounds=((0.5,3.0),), method='Nelder-Mead')
        rmin = res.x[0]*self.sigma
        epseff = abs(self.calc_pot_divk(rmin))
        return rmin, epseff

    def _calc_sigmaeff(self):
        """Calculate effective sigma
        Args:
            T (optional): K
        Returns:
            sigmaeff (m)
        """
        sol = SciOpt.root(self.calc_pot_divk, x0=self.sigma)
        return float(sol.x)

    def _calc_zero_entropic_force(self, T):
        """Calculate the largest position where the entropic force (d/dr)*exp(-beta*u(r))*r**2 is zero.
        Args:
            T: temperature [eps]
        Returns:
            sigma_s [sigma]
        """
        def fun(r):
            return np.exp(-self.calc_pot_divk(r)/T)*r*(2-self.calc_pot_divk(r)/T)
        sol1 = SciOpt.root(self.calc_pot_divk, x0=self.sigma)
        sol2 = SciOpt.root(self.calc_pot_divk, x0=self.sigma*1.6)
        return float(sol1.x), float(sol2.x)

    def calc_alpha(self, x=1):
        """Calculates effective alpha"""
        def integrand(r):
            return -r**2 * self.calc_pot_divk(r)

        normalization = self.epsdivkeff*self.sigmaeff**3
        integral, error = SciInt.quad(func=integrand, a=x*self.sigmaeff, b=self.rc,
                                      epsabs=1e-8*normalization, limit=10000)
        alphaeff = integral/normalization
        assert alphaeff>0
        assert error/alphaeff<1e-4, error
        return alphaeff

    def calc_B2(self, T, lowlim=0):
        '''Calculate second virial coefficient B2.
        Input:
            T: temperature [K]
            lowlim: lower integration limit [sigma]
        Returns:
            B2 [sigma^3].'''

        
        def fmayer(r,T):
            if r<0.3:
                return -1
            else:
                return np.exp(-self.calc_pot_divk(r)/T)-1

        def integrand_B2(r):
            return fmayer(r,T)*r**2

        # Integration using the substitution w = r/sigma -> dr = sigma*dw
        hs_contrib = 0
        if self.hardcore:
            hs_contrib = 2*np.pi/3 * self.sigma**3
            lowlim = self.sigma

        integral, error = SciInt.quad(func=integrand_B2, a=lowlim, b=self.rc,
            epsabs=1e-100, limit=10000)
        assert abs(error/integral)<1e-4, (error, integral)
        B2 = -2*np.pi*integral + hs_contrib
        return B2

    def calc_B3(self, T):
        '''Calculate third virial coefficient B3.
        Input:
            T [K]
        Returns:
            B3 [m^6].'''
        import math
        if self.hardcore:
            raise NotImplementedError

        def fmayer(r,T):
            if r<0.3:
                return -1
            else:
                return np.exp(-self.calc_pot_divk(r)/T)-1

        prefac = -8*np.pi**2/3.0
        def integrand_B3(h, x, r1):
            h2 = h*h
            x2 = x*x
            xmr2 = (x-r1)*(x-r1)
            r2 = math.sqrt(x2+h2)
            r12 = math.sqrt(xmr2+h2)
            integrand = fmayer(r1,T)*fmayer(r2,T)*fmayer(r12,T)*(r1*r1*h)
            return integrand
 
        lowlim = 0

        rc = self.rc
        rc2 = rc*rc

        r1min, r1max = 0, rc

        xmin = lambda r1: r1-rc
        xmax = lambda r1: rc

        hmin = 0
        hmax = lambda r1,x: math.sqrt( rc2 - max(abs(x),abs(x-r1))**2 )
        
        integral, error = SciInt.tplquad(func=integrand_B3,
                                         a=r1min,   b=r1max,
                                         gfun=xmin, hfun=xmax,
                                         qfun=hmin, rfun=hmax)
        print ("error, integral, ratio", error, integral, error/integral)
        assert abs(error/(abs(integral)+1e-4))<1e-3, (error, integral)
        B3 = prefac*integral
        return B3

    def calc_stickyness_tau(self, T):
        '''B2star = 1-1/(4*tau), see paragraph following Eq. 6 in Noro-Frenkel paper.'''
        B2star = self.calc_B2(T)/(2*np.pi*calc_dhs_rep(self, T)**3 /3)
        #B2star = self.calc_B2(T)/(2*np.pi*self.sigma**3 /3)
        tau = (4*(1-B2star))**(-1)
        return tau

    def effective_R(self, T):
        '''Eq. 8 in Noro-Frenkel paper.'''
        tau = self.calc_stickyness_tau(T)
        lam = (1+1/(4*tau*(np.exp(1/T)-1)))**(1/3)
        R = lam-1
        return R

    def plot(self, n=500, rmax=4, filename=None, T=None, kwargs=None, show=True):
        """Plot dimensionless potential as a function of separation
        Arguments
            n: Number of points plotted
            filename: Save plot to filename
        """
        kwargs = {} if kwargs is None else kwargs
        rvec = np.linspace(0.9,rmax,n)*self.sigma
        u = [self.calc_pot_divk(r)/self.epsdivk for r in rvec]
        plt.plot(rvec/self.sigma, u, **kwargs)
        plt.xlabel(r"$r/\sigma$")
        plt.ylabel(r"$u/\epsilon$")
        plt.xlim(xmax=rmax)
        plt.ylim(ymin=-1, ymax=1)
        plt.grid()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()


class SutherlandSum(RadiallySymmetricPotential):
    """A minimal implementation of the Sutherland sum potential family."""
    def __init__(self, sigma=1, epsdivk=1, C=(4,-4), lam=(12,6), **kwargs):
        self.C = C
        self.lam = lam
        self.nt = len(C)
        super().__init__(sigma=sigma, epsdivk=epsdivk, **kwargs)

    def calc_pot_divk(self, r):
        """Calculate potential
        Args:
            r: separation (m)
        Returns:
            potential value (K)
        """
        pot = 0
        for c, lam in zip(self.C, self.lam):
            pot = pot + c*(self.sigma/r)**lam
        return pot

    def report_coefficients(self):
        '''The Sutherland-sum representation'''

        print (f"{'term':10s} {'C':10s} {'exponent':10s}")
        for k in range(self.nt):
            print (f"{k:10d} {self.C(k):10.5f} {self.lam(k):10.5f}")
        print ("epsdivk/K {self.epsdivk}")
        print ("sigma/m   {self.sigma}")


class Mie(SutherlandSum):
    """The Mie potential."""
    def __init__(self, sigma=1, epsdivk=1, lama=6, lamr=12, **kwargs):
        """Set Mie-FH potential parameters (defaults to Lennard-Jones potential)
        Args:
        sigma - Position of zero potential (m)
        epsdivk - Potential depth (K)
        lama - Attraction exponent (-)
        lamr - Repulsion exponent (-)
        """
        # The Mie potential
        self.lama = lama
        self.lamr = lamr
        denom = lamr - lama
        exponent = lama/denom
        self.Cmie = lamr/denom*(lamr/lama)**exponent
        super().__init__(sigma, epsdivk, C=[self.Cmie, -self.Cmie], lam=[lamr, lama], **kwargs)


class MieQuadrupolar(RadiallySymmetricPotential):
    """The Mie potential with spherically averaged quadrupole moment."""
    def __init__(self, sigma=3.785e-10, epsdivk=253, lama=6, lamr=12, Q=-4.3, **kwargs):
        """Set parameters for a spherical molecule with a dipole or quadrupole moment
        Args:
        sigma - Position of zero potential (m)
        epsdivk - Potential depth (K)
        lama - Attraction exponent
        lamr - Repulsion exponent
        Q - Quadrupole moment (D*Å)
        """
        # The Mie potential
        super().__init__(sigma, epsdivk, **kwargs)
        self.lama = lama
        self.lamr = lamr
        self.Q = Q

        # Number of Sutherland terms
        self.nt = 3

        # Distance exponents
        self.lam = [lamr, lama, 10]

        # Beta exponents
        self.bexp = [0,0,1]

        # Adimensional prefactor to eps/(kB*T) term
        Q_SI = Q*DEBYE*1e-10
        Cq = - 1.4*(Q_SI*Q_SI/(epsdivk*KB*sigma**5))**2

        # Sutherland coefficients
        denom = (self.lamr - self.lama)
        exponent = self.lama/denom
        Cmie = self.lamr/denom*(self.lamr/self.lama)**exponent
        self.C = [Cmie, -Cmie, Cq]

    def calc_pot_divk(self, r):
        """Calculate potential
        Args:
            r: separation (m)
        Returns:
            potential value (K)
        """
        pot = 0
        for c, lam in zip(self.C, self.lam):
            pot = pot + c*(self.sigma/r)**lam
        return pot

class MieFH(RadiallySymmetricPotential):
    """The Mie-FH potential."""
    def __init__(self, sigma=1, epsdivk=1, lama=6, lamr=12, fh=0, redmass=None, T=None, **kwargs):
        """Set Mie-FH potential parameters (defaults to Lennard-Jones potential)
        Args:
        sigma - Position of zero potential (m)
        epsdivk - Potential depth (K)
        lama - Attraction exponent
        lamr - Repulsion exponent
        fh - order of FH correction
        redmass - reduced mass of the two interacting particles (kg)
        """
        # The Mie potential
        self.lama = lama
        self.lamr = lamr
        denom = (self.lamr - self.lama)
        exponent = self.lama/denom
        self.C = self.lamr/denom*(self.lamr/self.lama)**exponent

        # Feynman--Hibbs corrections
        self.fh = fh
        if fh>=1:
            assert T>0, T
            self.update_T(T)
            self.D = (H_PLANCK**2)/(96*KB*redmass*np.pi**2)
            self.Q1r = self.lamr*(self.lamr-1)
            self.Q2r = (self.lamr+2)*(self.lamr+1)*self.lamr*(self.lamr-1)
            self.Q1a = self.lama*(self.lama-1)
            self.Q2a = (self.lama+2)*(self.lama+1)*self.lama*(self.lama-1)

        super().__init__(sigma, epsdivk, **kwargs)

    def calc_pot_divk(self, r):
        """Calculate Mie potential
        Args:
        lamr - Repulsion exponent
        Returns:
        Mie potential (J)
        """
        urep = (self.sigma/r)**self.lamr
        uatt = (self.sigma/r)**self.lama

        u = urep - uatt
        if self.fh>=1:
            D_T = self.D/self.T
            en_r2 = 1.0/(r*r)
            u += (self.Q1r*urep-self.Q1a*uatt)*en_r2*D_T
        if self.fh>=2:
            u += (self.Q2r*urep-self.Q2a*uatt)*(en_r2*D_T)**2

        u = u*self.C*self.epsdivk
        return u

    @staticmethod
    def init_H2_MieFH1(T=300):
        return MieFH(sigma=3.0243e-10, epsdivk=26.706, lama=6, lamr=9, fh=1, redmass=MASSDICT["H2"]/2, T=T)

    def update_T(self, T):
        self.T = T

    def report_Sutherland_coefficients(self):
        '''The Sutherland-sum representation'''
        print ("- epsdivk (K)")
        print (self.epsdivk)
        print ("- sigma (m)")
        print (self.sigma)

        print ("- exponents (-)")
        print (self.lamr, self.lama, end=' ')
        if self.fh>=1:
            print (self.lamr+2, self.lama+2, end=' ')
        if self.fh>=2:
            print (self.lamr+4, self.lama+4, end=' ')
        print ()

        print ("- coefficients (-)")
        print (self.C, -self.C, end=' ')
        if self.fh>=1:
            print (self.C*self.D*self.Q1r/self.sigma**2/self.epsdivk, -self.C*self.D*self.Q1a/self.sigma**2/self.epsdivk, end=' ')
        if self.fh==2:
            print (self.C*self.D**2 * self.Q2r/self.sigma**4, -self.C*self.D**2 * self.Q2a/self.sigma**4, end=' ')
        print ()

        print ("- exponents for (epsdivk/T) (-)")
        print (0, 0, end=' ')
        if self.fh>=1:
            print (1, 1, end=' ')
        if self.fh==2:
            print (2, 2, end=' ')
        print ()

    def set_deboer(self, Lambda):
        '''Re-compute the Feynman--Hibbs D parameter from the de Boer parameter'''
        self.D = self.epsdivk*(Lambda*self.sigma)**2/(48*KB*np.pi**2)


class Yukawa(RadiallySymmetricPotential):
    """The Yukawa potential."""
    def __init__(self, sigma=1, epsdivk=1, lam=1):
        """Set Yukawa potential parameters
        Args:
        sigma - Hard-core diameter (m)
        epsdivk - Potential depth (K)
        lam - range (-)
        """
        assert sigma>0
        assert epsdivk>0
        assert lam>0
        self.lam = lam
        super().__init__(sigma, epsdivk, hardcore=True, precalc=False)

    def calc_pot_divk(self, r):
        """Calculate potential
        """
        if r < self.sigma:
            return np.inf
        else:
            rdivsigma = r/self.sigma
            return -self.epsdivk*np.exp(self.lam*(1-rdivsigma))/(rdivsigma)


class SquareWell(RadiallySymmetricPotential):
    """The square-well potential."""
    def __init__(self, sigma=1, epsdivk=1, lam=1.542):
        """Set Yukawa potential parameters (default to SW potential with same alpha as LJ).
        One obtains the hard-sphere potential by setting lam=0
        Args:
        sigma - Hard-core diameter (m)
        epsdivk - Potential depth (K)
        lam - range (-)
        """
        self.hardcore = True
        self.lam = lam
        self.rmin = sigma
        super().__init__(sigma, epsdivk, hardcore=True, precalc=False)

    def calc_pot_divk(self, r):
        """Calculate SW potential
        """
        if r < self.sigma:
            return np.inf
        elif r<self.lam*self.sigma:
            return -self.epsdivk
        else:
            return 0

    def calc_alpha_x(self, x=1):
        if x<1:
            raise ValueError()
        return (self.lam**3-min(x,self.lam)**3)/3

    def calc_B2(self, T):
        B2 = 2*np.pi*self.sigma**3 / 3
        if self.lam>1:
            B2 += (2*np.pi*self.sigma**3/3)*(1-np.exp(self.epsdivk/T))*(self.lam**3-1)
        return B2


class LJSpline(RadiallySymmetricPotential):
    """The Lennard-Jones-spline potential."""
    def __init__(self, sigma=1, epsdivk=1):
        """Set LJs potential parameters
        Args:
        sigma - diameter (m)
        epsdivk - potential depth (K)
        """
        self.RS = (26/7.0)**(1/6.0)
        self.rc = 67*self.RS/48.0
        self.A = -24192*epsdivk/(3211*self.RS**2)
        self.B = -387072*epsdivk/(61009*self.RS**3)
        super().__init__(sigma, epsdivk, rc=self.rc)

    def calc_pot_divk(self, r):
        """Calculate potential
        """
        if 0 <= r <= self.RS:
            six = (self.sigma/r)**6
            return 4*self.epsdivk*six*(six-1)
        elif r <= self.rc:
            return (r-self.rc)**2 * (self.A + self.B*(r-self.rc))
        else:
            return 0.0


class Wang2020Potential(RadiallySymmetricPotential):
    """The LJ-like potential by Wang et al. 2020 (10.1039/C9CP05445F)"""
    def __init__(self, sigma=1, epsdivk=1):
        """Set potential parameters
        Args:
        sigma - diameter (m)
        epsdivk - potential depth (K)
        """
        self.rc = 2
        super().__init__(sigma, epsdivk, rc=self.rc)

    def calc_pot_divk(self, r):
        """Calculate potential
        """
        if r <= self.rc:
            rc, sig = self.rc, self.sigma
            rcdivsig2 = (rc/sig)**2
            alpha = 2*rcdivsig2 * (3 /(2*rcdivsig2 - 2))**3
            rm = rc*(3/(1 + 2*rcdivsig2))**0.5
            return self.epsdivk*alpha * ((sig/r)**2-1) * ((rc/r)**2-1)**2
        else:
            return 0.0

class InversePowerPotential(RadiallySymmetricPotential):
    """Inverse power potential, i.e. a (soft) Sutherland potential."""
    def __init__(self, sigma=1, epsdivk=1, lam=12):
        # The Mie potential
        self.lam = lam
        super().__init__(sigma, epsdivk)

    def calc_pot_divk(self, r):
        if r<=1e-10*self.sigma:
            return np.inf
        else:
            return self.epsdivk*(self.sigma/r)**self.lam

    def precalc(self,):
        self.rmin = np.inf
        self.epsdivkeff = 0.0
        self.sigmaeff = 0.0
        return

if __name__=="__main__":

    # M = MieQuadrupolar()
    # print (M.C)

    # # Display Sutherland-Sum representation for Mie-FH1 representation of H2
    # M = MieFH.init_H2_MieFH1(T=20)

    LJ = Mie(sigma=1, epsdivk=1)
    # print (LJ.calc_B2(0.75))

    LJ.plot(show=False)
    rvec = np.linspace(0.8, 4, 500)*LJ.sigma
    dudrvec  = [LJ.calc_pot_derivative(r) for r in rvec]
    #T = 1.31
    #probvec = [np.exp(-LJ.calc_pot_divk(r)/T)*r*r for r in rvec]

    plt.plot(rvec, dudrvec)
    plt.ylim(-1,5)
    plt.show()
    # def plot(self, n=500, rmax=4, filename=None, T=None, kwargs=None, show=True):
    #     """Plot dimensionless potential as a function of separation
    #     Arguments
    #         n: Number of points plotted
    #         filename: Save plot to filename
    #     """
    #     kwargs = {} if kwargs is None else kwargs
    #     rvec = np.linspace(0.9, rmax, n)*self.sigma
    #     u = [self.calc_pot_divk(r)/self.epsdivk for r in rvec]
    #     plt.plot(rvec/self.sigma, u, **kwargs)
    #     plt.xlabel(r"$r/\sigma$")
    #     plt.ylabel(r"$u/\epsilon$")
    #     plt.xlim(xmax=rmax)
    #     plt.ylim(ymin=-1, ymax=1)
    #     plt.grid()
    #     if filename is not None:
    #         plt.savefig(filename)
    #     if show:
    #         plt.show()
