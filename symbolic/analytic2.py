import sympy as sp
from sympy import exp, log, pi, sympify
import numpy as np
mx, my, ux, uy, vx, vy = sp.symbols('mx my ux uy vx vy')

# s1, s2 = sp.symbols('s1 s2', positive=True)
s1, s2 = sp.symbols('s1 s2', positive=True)

v1 = sp.Matrix([vx, vy]) # direction vector, first eignevector of the covariance matrix
v2 = sp.Matrix([-vy, vx]) # second eigenvector
          
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
    return 1 / (1 + exp(-sympify('1.6') * z - sympify('0.07') * z**3))

def S_pixel(x, sigma):
    """ Evaluate the integral approximation of the gaussian cdf between x + step and x - step """
    return S(x + sp.sympify('0.5'), sigma) - S(x - sp.sympify('0.5'), sigma) 

i1 = s1 * S_pixel(tx, s1)
i2 = s2 * S_pixel(ty, s2)
p = 2 * sp.pi * i1 * i2





vars = dict(vx=vx, vy=vy, mx=mx, my=my, s1=s1, s2=s2)

cse, derivs = sp.cse([sp.diff(p, i) for i in vars.values()] + [p],  optimizations='basic')
derivs = {k:v for k, v in zip(vars.keys(), derivs)}

def rev_cse(expr):
  for (s, sub_expr) in reversed(cse):
    expr = expr.subs(s, sub_expr)
  return expr

expanded = {k:rev_cse(deriv) for k, deriv in derivs.items()}

for k, v in cse:
    print(f"{k} = {v}")

for k, v in derivs.items():
    print(f"{k} = {v}")




x0 = mx - ux
x1 = 1/s1
x2 = my - uy
x3 = vx*x0 + vy*x2
x4 = x3 + 0.5
x5 = x4**2
x6 = s1**(-2)
x7 = 0.07*x6
x8 = exp(x1*x4*(x5*x7 + 1.6)) + 1
x9 = 0.21*x6
x10 = -x0
x11 = -x2
x12 = vx*x10 + vy*x11
x13 = x12 - 0.5
x14 = exp(-x1*x13*(x13**2*x7 + 1.6))
x15 = x14*(x5*x9 + 1.6)
x16 = 0.5 - x3
x17 = x16**2
x18 = 1 + exp(-x1*x16*(x17*x7 + 1.6))
x19 = x12 + 0.5
x20 = exp(-x1*x19*(x19**2*x7 + 1.6))
x21 = x20*(x17*x9 + 1.6)
x22 = 1/s2
x23 = vy*x10
x24 = vx*x11
x25 = x23 - x24 + 0.5
x26 = s2**(-2)
x27 = x25**2*x26
x28 = exp(x22*x25*(0.07*x27 + 1.6)) + 1
x29 = -vx*x2 + vy*x0 + 0.5
x30 = x29**2
x31 = 0.07*x26
x32 = 1 + exp(-x22*x29*(x30*x31 + 1.6))
x33 = -1/x32 + 1/x28
x34 = -s2*x33
x35 = x34*(x15/x8**2 - x21/x18**2)
x36 = -x25
x37 = exp(-x22*x36*(x31*x36**2 + 1.6))
x38 = x37*(0.21*x27 + 1.6)
x39 = -x23 + x24 + 0.5
x40 = exp(-x22*x39*(x31*x39**2 + 1.6))
x41 = x40*(0.21*x26*x30 + 1.6)
x42 = -x41/x32**2 + x38/x28**2
x43 = 1/x8 - 1/x18
x44 = -s1*x43
x45 = 2*pi
x46 = -x42*x44
x47 = x15/(x14 + 1)**2
x48 = x21/(x20 + 1)**2
x49 = x34*(-x47 + x48)
x50 = x41/(x40 + 1)**2
x51 = x38/(x37 + 1)**2
x52 = x34*x45

derived = dict(
  vx = x45*(x0*x35 + x2*x42*x44),
  vy = x45*(x0*x46 + x2*x35),
  mx = x45*(-vx*x49 + vy*x46),
  my = -x45*(vx*x44*(x50 - x51) + vy*x49),
  s1 = x52*(-x1*(x16*x48 + x4*x47) - x43),
  s2 = x44*x45*(-x22*(x25*x51 + x29*x50) - x33)
)



   

def equal(a, b):
  return sp.nsimplify(a - b, rational=True) == 0

for k, deriv in derivs.items():
    print(k, equal(derivs[k], derived[k]))