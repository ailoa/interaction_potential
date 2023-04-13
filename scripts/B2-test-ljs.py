# Compare second virial expansion with supersaturated gas simulations
# for the LJ-spline potential
import sys; sys.path.append('../src/')
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import N_A as NA, k as KB
from thermopack.ljs_wca import ljs_uv

# # Avogadros number
# NA = 6.02214076e23
# KB = 1.380650524e-23

# Initialize uv-theory eos
ljs = ljs_uv()
sigma, eps = ljs.get_sigma_eps()

def calc_reduced_P(Pa, eps, sigma):
    """ Calculate reduced pressure
    """
    Pstar = np.zeros_like(Pa)
    Pstar = Pa*sigma**3/eps/KB
    return Pstar

def calc_reduced_T(Ta, eps):
    """ Calculate reduced temperature
    """
    Tstar = np.zeros_like(Ta)
    Tstar = Ta/eps
    return Tstar

def calc_reduced_rho(rhoa, sigma):
    """ Calculate reduced density
    """
    rhoStar = np.zeros_like(rhoa)
    rhoStar = sigma**3*NA*rhoa
    return rhoStar

def calc_real_rho(rhoStar, sigma):
    """ Calculate density from reduced density
    """
    rhoa = np.zeros_like(rhoStar)
    rhoa = rhoStar/(sigma**3*NA)
    return rhoa

# States to examine
z = [1.0]
Tvec = [0.65, 0.70, 0.75, 0.8, 0.85]
rhovec = np.linspace(1e-6, 0.2)
rhorealvec = calc_real_rho(rhovec, sigma)

# Initialize virial eos
from radially_sym_pot import LJSpline
pot = LJSpline()
def P_B2(T, rho):
    return rho*T*(1+pot.calc_B2(T)*rho)

# Compare second virial expansion with simulation data and LJs EoS
M = np.genfromtxt('../data/IsothermsMetaStable_MD_ElongatedBox.txt')
T_sim, rho_sim, P_sim = M[:,0], M[:,1], M[:,2]
for T in Tvec:
    idcs = T_sim==T
    plt.scatter(rho_sim[idcs], P_sim[idcs], zorder=10, s=30)
    pvec_uv = [calc_reduced_P(ljs.pressure_tv(T*eps, 1/rho, z)[0], eps, sigma) for rho in rhorealvec]
    pvec_b2 = [P_B2(T=T, rho=rho) for rho in rhovec]
    plt.plot(rhovec, pvec_uv, ls='-', color='orange')
    plt.plot(rhovec, pvec_b2, ls='--', color='blue')
    Psatreal = ljs.dew_pressure(T*eps, z)[0]
    vreal = ljs.specific_volume(T*eps, Psatreal, z, phase=ljs.VAPPH)[0]
    plt.scatter([calc_reduced_rho(1/vreal, sigma)], [calc_reduced_P(Psatreal, eps, sigma)], s=20, marker='d', color='cyan', zorder=100)
plt.grid(); plt.xlim(0,0.2); plt.ylim(0,0.08)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$P$')
plt.savefig("B2-hypothesis_ljs.pdf")
plt.show()
