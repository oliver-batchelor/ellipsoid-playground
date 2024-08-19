import symengine as sp
from symengine import exp, log, pi, sympify

from util import multi_num_equal, num_equal

mx, my, ux, uy, vx, vy = sp.symbols('mx my ux uy vx vy', real=True)

# s1, s2 = sp.symbols('s1 s2', positive=True)
s1, s2 = sp.symbols('s1 s2', positive=True, real=True)

def perp(v):
    return sp.Matrix([-v[1], v[0]])

v1 = sp.Matrix([vx, vy]) # direction vector, first eignevector of the covariance matrix
v2 = perp(v1) # second eigenvector

# v2 = sp.Matrix([-vy, vx]) # second eigenvector
          
u = sp.Matrix([ux, uy]) # pixel centre `u`` in paper
m = sp.Matrix([mx, my]) # gaussian mean - \hat{u} in paper


# local coordinate of pixel centre in gaussian coordinate system
# \tilde{u} in paper (tx, ty)
d = u - m

tx = d.dot(v1)
ty = d.dot(v2)

def S(x, sigma=1):
    """ Approximate gaussian cdf """
    z = x / sigma
    return 1 / (1 + exp(-1.6 * z - 0.07 * z**3))

def S_pixel(x, sigma):
    """ Evaluate the integral approximation of the gaussian cdf between x + step and x - step """
    return S(x + 0.5, sigma) - S(x - 0.5, sigma) 

i1 = s1 * S_pixel(tx, s1)
i2 = s2 * S_pixel(ty, s2)
i2d = 2 * sp.pi * i1 * i2



def dS(x, sigma=1):
    """ Derivative of the approx gaussian cdf S at x """
    sx = S(x, sigma)
    return (1.6 + 0.21 * (x/sigma)**2) * sx * (1 - sx)


def dS_dx(x, sigma=1):
    return dS(x, sigma) * 1/sigma

def dS_dsigma(x, sigma=1):
  return dS(x, sigma) * -x/(sigma ** 2)

                                
x, s = sp.symbols('x s')
num_equal("S_pixel_x", sp.diff(S_pixel(x, s), x),  dS_dx(x + 0.5, s) - dS_dx(x - 0.5, s))

num_equal("S_pixel_sigma", sp.diff(S_pixel(x, s), s),  dS_dsigma(x + 0.5, s) - dS_dsigma(x - 0.5, s))


di_dMean = (dS_dx(tx + 0.5, s1) - dS_dx(tx - 0.5, s1)) * s1 *  -v1
multi_num_equal("i1_m", (sp.diff(i1, mx), sp.diff(i1, my)), di_dMean)


di_dMean = 2 * pi * (i2  * ((dS_dx(tx + 0.5, s1) - dS_dx(tx - 0.5, s1)) * s1 * -v1)
                    + i1 * ((dS_dx(ty + 0.5, s2) - dS_dx(ty - 0.5, s2)) * s2 * -v2))


multi_num_equal("di_dMean", (sp.diff(i2d, mx), sp.diff(i2d, my)), di_dMean)


di_s1 = 2 * pi * i2 * ( S_pixel(tx, s1)
                  + s1 * (dS_dsigma(tx + 0.5, s1) - dS_dsigma(tx - 0.5, s1)))

di_s2 = 2 * pi * i1 * ( S_pixel(ty, s2)
                  + s2 * (dS_dsigma(ty + 0.5, s2) - dS_dsigma(ty - 0.5, s2)))


num_equal("i2d_s1", sp.diff(i2d, s1), di_s1)
num_equal("i2d_s2", sp.diff(i2d, s2), di_s2)



di_dv1 = 2 * pi * (i2 * s1  * (dS_dx(tx + 0.5, s1) - dS_dx(tx - 0.5, s1)) * d          # gradient on first eigenvector (v1)
                +  i1 * s2  * (dS_dx(ty + 0.5, s2) - dS_dx(ty - 0.5, s2)) * -perp(d))  # gradient on second eigenvector (v2 = perp(v1))


multi_num_equal("i2d_v1", (sp.diff(i2d, vx), sp.diff(i2d, vy)), di_dv1)
