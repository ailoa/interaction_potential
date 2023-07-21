# Compare second virial expansion with supersaturated gas simulations
# for the LJ-spline potential
import sys; sys.path.append('../src/')
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import N_A as NA, k as KB
from thermopack.ljs_wca import ljs_uv

# Avogadros number
NA = 6.02214076e23
KB = 1.380650524e-23

print (NA*KB)
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
Tvec = [0.55, 0.60, 0.65]
rhovec = np.linspace(1e-6, 0.333*0.5, 10)
rhorealvec = calc_real_rho(rhovec, sigma)

# Initialize virial eos
from radially_sym_pot import LJSpline
pot = LJSpline()

# Calculate B2 and B3 for all temperatures
B2dict = {0.65: -7.416496183392034, 0.7: -6.34324042988608, 0.75: -5.484536222798156, 0.8: -4.783514885149114, 0.85: -4.201456718097707, 0.55: -10.6044916636537, 0.6: -8.790942475384952}
B3dict = {0.65: -4.519543534392317, 0.7: -0.07311807802446374, 0.75: 2.072207939706353, 0.8: 3.0670265143300237, 0.85: 3.4721746620620215, 0.55: -33.55494429737849, 0.6: -13.757102776313362}
for t in Tvec:
    if not t in B2dict:
        B2dict[t] = pot.calc_B2(t)
    if not t in B3dict:
        B3dict[t] = pot.calc_B3(t)
        
for t in Tvec:
    B2_eos, B3_eos = ljs.virial_coeffcients(t*eps, n=[1.0])
    B2dict[t] = B2_eos/(NA*sigma**3)
    B3dict[t] = B3_eos/(NA*sigma**3)**2
    print (t)
    print (f"B2_eos({t}) {B2_eos/(NA*sigma**3)}")
    print (f"B2_pot({t}) {B2dict[t]}")
    print (f"B3_eos({t}) {B3_eos/(NA*sigma**3)**2}")
    print (f"B3_pot({t}) {B3dict[t]}")
    
def P_B2(T, rho):
    return rho*T*(1 + B2dict[T]*rho)

def P_B3(T, rho):
    return rho*T*(1 + B2dict[T]*rho + B3dict[T]*rho**2)

T = 0.7
print (B2dict[T]*rhovec * 100)
print (B3dict[T]*rhovec**2 * 100)
print (B3dict[T]*rhovec**2 /(B2dict[T]*rhovec) * 100)


# Compare second virial expansion with simulation data and LJs EoS
M = np.genfromtxt('../data/IsothermsMetaStable_MD_ElongatedBox.txt')
T_sim, rho_sim, P_sim = M[:,0], M[:,1], M[:,2]
rvec = np.linspace(0,0.3, 1000)
for T in Tvec:
    idcs = T_sim==T
    plt.scatter(rho_sim[idcs], P_sim[idcs], zorder=10, s=30)
    pvec_uv = [calc_reduced_P(ljs.pressure_tv(T*eps, 1/rho, z)[0], eps, sigma) for rho in rhorealvec]
    plt.plot(rhovec, pvec_uv, ls='-', color='orange')
    pvec_b2 = [P_B2(T=T, rho=rho) for rho in rvec]
    pvec_b3 = [P_B3(T=T, rho=rho) for rho in rvec]
    plt.plot(rvec, pvec_b2, ls='--', color='blue')
    plt.plot(rvec, pvec_b3, ls='--', color='navy')
    Psatreal = ljs.dew_pressure(T*eps, z)[0]
    vreal = ljs.specific_volume(T*eps, Psatreal, z, phase=ljs.VAPPH)[0]
    plt.scatter([calc_reduced_rho(1/vreal, sigma)], [calc_reduced_P(Psatreal, eps, sigma)], s=20, marker='d', color='cyan', zorder=100)
plt.grid(); plt.xlim(0,0.2); plt.ylim(0,0.08)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$P$')
#plt.savefig("B2-hypothesis_ljs.pdf")
plt.show()
