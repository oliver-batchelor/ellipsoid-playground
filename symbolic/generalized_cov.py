import sympy as sp
from sympy import exp, log
x, y, u, v, a, b, c, B = sp.symbols('x y u v a b c B')

dx = x - u
dy = y - v

inner =  (0.5 * (dx**2 * a + dy**2 * c) + dx * dy * b) 
p = exp( -(inner ** B) )

cse, derivs = sp.cse([sp.diff(p, i) for i in [u, v, a, b, c, B]] + [p],  optimizations='basic')

print(dict(cse))
print(derivs)

def rev_cse(expr):
  for (s, sub_expr) in reversed(cse):
    expr = expr.subs(s, sub_expr)
  return expr

expanded = [rev_cse(deriv) for deriv in derivs]


d_inner = B * (inner ** (B - 1))  * p

du = (a*dx + b*dy) * d_inner
dv = (c*dy + b*dx) * d_inner


da = -0.5 * dx**2 * d_inner
db = -dx * dy * d_inner
dc = -0.5* dy**2 * d_inner

dB = -inner ** B * log(inner) * p

def equal(a, b):
  return sp.simplify(a - b) == 0

print([equal(sp.diff(p, var), deriv) 
  for var, deriv in zip([u, v, a, b, c, B], [du, dv, da, db, dc, dB])  ])


