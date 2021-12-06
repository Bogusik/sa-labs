from numpy.polynomial import Polynomial as pm
import numpy as np
from scipy import special
def basis_sh_chebyshev1(degree):
    basis = [pm([1]), pm([0, 1])]
    for i in range(degree):
        basis.append(pm([0, 2])*basis[-1] - basis[-2])
    return basis


def basis_sh_chebyshev2(degree):
    basis = [pm([1]), pm([-2, 4])]
    for i in range(degree):
        basis.append(pm([-2, 4])*basis[-1] - basis[-2])
    return basis


def basis_sh_legendre(degree):
    basis = [pm([1])]
    for i in range(degree):
        if i == 0:
            basis.append(pm([-1, 2]))
            continue
        basis.append((pm([-2*i - 1, 4*i + 2])*basis[-1] - i * basis[-2]) / (i + 1))
    return basis


def basis_hermite(degree):
    basis = [pm([0]), pm([1])]
    for i in range(degree):
        basis.append(pm([0,2])*basis[-1] - 2 * i * basis[-2])
    return basis


def basis_laguerre(degree):
    basis = [pm([1])]
    for i in range(degree):
        if i == 0:
            basis.append(pm([1, -1]))
            continue
        basis.append(pm([2*i + 1, -1])*basis[-1] - i * i * basis[-2])
    return basis


def polynom_coef_resolver(poly_type,degree):
    if poly_type =='chebyshev':
        return basis_sh_chebyshev1(degree)[degree].coef
    elif  poly_type =='chebyshev_2_type':
        return basis_sh_chebyshev2(degree)[degree].coef
    elif poly_type  == 'legandre':
        return basis_sh_legendre(degree)[degree].coef
    elif  poly_type == 'laguerre':
        return basis_laguerre(degree)[degree].coef
    elif   poly_type == 'hermite':
        return basis_hermite(degree)[degree].coef

def transform_to_standard(poly_type,degree,coef):
        
    std_coeffs = polynom_coef_resolver(poly_type,degree)*coef
    res_string=''
    for i in range(len(std_coeffs)):
        if std_coeffs[i]==0:
            continue
        res_string += '{0:+}'.format(std_coeffs[i]) + "x^{i}".format(i=i)
    return res_string        

 