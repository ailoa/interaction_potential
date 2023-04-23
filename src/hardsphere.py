import numpy as np

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
    # if r>1.5*dhs:
    #     return 0.0
    delta = dhs*1e-5
    numdiff = (y_hs_desousa_amotz(r, rho, dhs+delta) - y_hs_desousa_amotz(r, rho, dhs-delta))/(2*delta)
    return numdiff
    # eta = np.pi / 6 * rho * dhs**3
    # denom = (1.0 - eta)**3
    # A = (3.0 - eta) / denom - 3.0
    # B = -3 * eta * (2.0 - eta) / denom
    # C = np.log((2.0 - eta) * (2 * denom)) - eta * (2.0 - 6 * eta + 3 * eta**2) / denom
    # y = np.exp(A + B * (r / dhs) + C * (r / dhs)**3)

    # deta_ddhs = np.pi / 2 * rho * dhs**2
    # ddenom_ddhs = -3 * (1.0 - eta)**2 * deta_ddhs

    # dA_ddhs = ((-1) * denom * deta_ddhs - (3.0 - eta) * ddenom_ddhs) / denom**2
    # dB_ddhs = ((-3 * (2.0 - eta) * denom - (-3 * eta) * (-3 * (1.0 - eta)**2)) * deta_ddhs) / denom**2
    # dC_ddhs = (1 / ((2.0 - eta) * (2 * denom)) * (-4 * denom * deta_ddhs - (2.0 - eta) * ddenom_ddhs) -
    #           (deta_ddhs * (2.0 - 6 * eta + 3 * eta**2) / denom + eta * (2.0 - 6 * eta + 3 * eta**2) * ddenom_ddhs / denom**2))

    # dy_ddhs = y * (dA_ddhs + dB_ddhs * (r / dhs) - B * (r / dhs**2) + dC_ddhs * (r / dhs)**3 - 3 * C * (r / dhs**4))

    # return dy_ddhs


# r = 1.1
# rho = 0.2
# dhs = 0.98
# delta = 1e-10

# numdiff = (y_hs_desousa_amotz(r, rho, dhs+delta) - y_hs_desousa_amotz(r, rho, dhs-delta))/(2*delta)
# print (numdiff)
# print (dy_hs_desousa_amotz_ddhs(r, rho, dhs))
