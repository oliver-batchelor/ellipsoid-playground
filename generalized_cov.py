import sympy as sp
from sympy import exp, log
x, y, u, v, a, b, c, p, B = sp.symbols('x y u v a b c p B')

dx = x - u
dy = y - v

inner =  (0.5 * (dx**2 * a + dy**2 * c) - dx * dy * b) 
p = exp( -(inner ** B) )

print (sp.diff(p, u))


du = -B*(0.5*a*(2*u - 2*x) + b*dy) * (inner ** B)  * p / inner
dv = -B*(0.5*c*(2*v - 2*y) + b*dx) * (inner ** B)  * p / inner

print(sp.diff(p, u) == du)
print(sp.diff(p, v) == dv)



da = -0.5*B * dx**2 * (inner ** B) * p / inner
db = B * dx * dy * (inner ** B) * p / inner
dc = -0.5*B * dy**2 * (inner ** B) * p / inner

print(sp.diff(p, a) == da)
print(sp.diff(p, b) == db)
print(sp.diff(p, c) == dc)


dc = -inner ** B * log(inner) * p

print(sp.diff(p, B) == dc)

