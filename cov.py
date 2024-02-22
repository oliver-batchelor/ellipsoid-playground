import sympy as sp
from sympy import exp
x, y, u, v, a, b, c, p = sp.symbols('x y u v a b c p')


dx = x - u
dy = y - v


inner = 0.5 * (dx**2 * a + dy**2 * c) + dx * dy * b

p = exp(-inner)

du = (a*dx + b*dy)*p
dv = (b*dx + c*dy)*p

da = -0.5*dx**2*p
db = -dx*dy*p
dc = -0.5*dy**2*p

def equal(a, b):
  return sp.simplify(a - b) == 0

print([equal(sp.diff(p, var), deriv) 
  for var, deriv in zip([u, v, a, b, c], [du, dv, da, db, dc])  ])




