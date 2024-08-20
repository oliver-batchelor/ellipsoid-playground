import sympy as sp
from sympy import exp

x, y, u, v, a, b, c = sp.symbols('x y u v a b c')


dx = x - u
dy = y - v


inner = 0.5 * (dx**2 * a + dy**2 * c) + dx * dy * b

p = exp(-inner)

du = (a*dx + b*dy)*p
dv = (b*dx + c*dy)*p

da = -0.5*dx**2*p
db = -dx*dy*p
dc = -0.5*dy**2*p


cse, derivs = sp.cse([sp.diff(p, i) for i in [u, v, a, b, c]] + [p],  optimizations='basic')
print(dict(cse))



def equal(a, b):
  return sp.simplify(a - b) == 0

print([equal(sp.diff(p, var), deriv) 
  for var, deriv in zip([u, v, a, b, c], [du, dv, da, db, dc])  ])




