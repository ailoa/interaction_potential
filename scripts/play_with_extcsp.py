import sys; sys.path.append('../src/')
import matplotlib.pyplot as plt
import numpy as np
from radially_sym_pot import *
import scipy.optimize as SciOpt

# dhs_rep = sigma -> tau=0.0986 is optimal
# dhs_rep = NoroFrenkel -> tau=0.1 is optimal
def temperature_at_tau_spec(M, tau=0.098):
    def resid(T):
        return M.calc_stickyness_tau(T)-tau
    sol = SciOpt.root(resid, x0=0.6)
    return float(sol.x)


class PotentialData(object):
    def __init__(self):
        self.D = {}
    
    def register_potentials(self, name, potlist, Tclist):
        self.D[name] = {}
        self.D[name]['pot'] = potlist
        self.D[name]['Tc'] = Tclist

    def plot_csp(self):
        namelist = self.D.keys()
        print (f"{'NAME':15s}, {'TC':>10s}, {'R-EST':>10s}, {'R-ERR %':>10s}, {'ALPHA-EST':>10s} {'ALPHA-ERR %':>10s}")
        for name in namelist:
            tclist = self.D[name]['Tc']
            potlist = self.D[name]['pot']
            alphavec, Rvec, Rvectau01 = [], [], []
            Tattau01vec = []
            for pot, tc in zip(potlist, tclist):
                T_at_tau = temperature_at_tau_spec(pot)
                alphavec.append(pot.calc_alpha())
                Rvec.append(pot.effective_R(tc))
                Rvectau01.append(pot.effective_R(T_at_tau))
                Tattau01vec.append(T_at_tau)
                r_est = T_at_tau
                r_err = (r_est-tc)/tc * 100
                a_est = 0.254+1.173*alphavec[-1]
                a_err = (a_est-tc)/tc * 100
                print (f"{name:15s}, {tc:10.3f}, {r_est:10.3f}, {r_err:10.1f}, {a_est:10.3f}, {a_err:10.1f}")
                # print (T_at_tau/tc)
                # print ((0.26 + 2.1*pot.effective_R(T_at_tau)/tc))
                # print (0.26 + 2.1*Rvec[-1])
                # print (pot.calc_stickyness_tau(tc), Rvec[-1])
                # print ()
            plt.figure("alpha")
            plt.scatter(alphavec, tclist, label=name)
            # plt.figure("R")
            # plt.scatter(Rvec, tclist, label=name)
            plt.figure("Rtau01")
            plt.scatter(Tattau01vec, tclist, label=name)

        # # The Noro-Frenkel extended CSP
        # plt.figure("R")
        # Rvec = np.linspace(0,1)
        # plt.plot(Rvec, 0.26+2.1*Rvec, ls='-', zorder=-10)
        # plt.xlabel("R"); plt.ylabel("Tc*")

        # My proposed extended CSP
        plt.figure("Rtau01")
        Tvec = np.linspace(0,3)
        plt.plot(Tvec, Tvec, ls='-', zorder=-10)
        plt.xlabel("Tattau"); plt.ylabel("Tc*")
        
        # Ramrattan's alpha hypothesis
        plt.figure("alpha")
        alphavec = np.linspace(0,2.5)
        plt.plot(alphavec, 0.254+1.173*alphavec, ls='-', zorder=-10)
        plt.xlabel("alpha"); plt.ylabel("Tc*")
        
        plt.legend()
        plt.show()

# Register potential data
PD = PotentialData()
PD.register_potentials("SquareWell",
                       [SquareWell(lam=lam) for lam in (1.25, 1.375, 1.5, 1.75, 2)],
                       [0.78, 1.01, 1.218, 1.79, 2.61])
PD.register_potentials("Yukawa",
                        [Yukawa(lam=lam) for lam in (1.8,3,4,7)],
                        [1.170,0.715,0.576,0.412])
PD.register_potentials("Mie",
                       [MieFH(lama=n, lamr=2*n) for n in (6,7,8,9,11,12,18)],
                       [1.316,0.997,0.831,0.730,0.603,0.560,0.425])
PD.register_potentials("LJSpline",
                       [LJSpline()],
                       [0.885])
ljts_list = [MieFH(rc=2.5, shift=True), MieFH(rc=2**(7/6), shift=True)]
PD.register_potentials("LJCutShift",
                       ljts_list,
                       [1.086/ljts_list[0].epsdivkeff, 0.998/ljts_list[1].epsdivkeff])
ljt_list = [MieFH(rc=2, shift=False), MieFH(rc=2.5, shift=False), MieFH(rc=5, shift=False)]
PD.register_potentials("LJCut",
                       ljt_list,
                       [1.061, 1.1875, 1.281]) # 1&3:Panagiatapoulous, 2:Loscar

# Analyze
PD.plot_csp()
