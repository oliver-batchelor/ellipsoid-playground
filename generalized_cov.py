import sympy as sp
from sympy import exp, log
x, y, u, v, a, b, c, B = sp.symbols('x y u v a b c B')

dx = x - u
dy = y - v

inner =  (0.5 * (dx**2 * a + dy**2 * c) - dx * dy * b) 
p = exp( -(inner ** B) )

cse, derivs = sp.cse([sp.diff(p, i) for i in [u, v, a, b, c, B]] + [p],  optimizations='basic')

print(dict(cse))
print(derivs)

def rev_cse(expr):
  for (s, sub_expr) in reversed(cse):
    expr = expr.subs(s, sub_expr)
  return expr

expanded = [rev_cse(deriv) for deriv in derivs]

def equal(a, b):
  return sp.simplify(a - b) == 0

print(equal(expanded[-1], p))

du = -B*(a*(u - x) + b*dy) * (inner ** B)  * p / inner
dv = -B*(c*(v - y) + b*dx) * (inner ** B)  * p / inner

print(equal(sp.diff(p, u), du))
print(equal(sp.diff(p, v), dv))

da = -0.5*B * dx**2 * (inner ** B) * p / inner
db = B * dx * dy * (inner ** B) * p / inner
dc = -0.5*B * dy**2 * (inner ** B) * p / inner

print(sp.diff(p, a) == da)
print(sp.diff(p, b) == db)
print(sp.diff(p, c) == dc)


dc = -inner ** B * log(inner) * p

print(sp.diff(p, B) == dc)

