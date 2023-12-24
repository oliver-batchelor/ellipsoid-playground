import sympy as sp
from sympy import exp
x, y, u, v, a, b, c, p = sp.symbols('x y u v a b c p')


dx = x - u
dy = y - v


du = (sp.diff(p, u))
dv = (sp.diff(p, v))

print(du)
print(dv)

p = exp(-0.5 * (dx**2 * a + dy**2 * c) - dx * dy * b)

du = (-0.5*a*(2*u - 2*x) + b*dy)*p
dv = (b*dx - 0.5*c*(2*v - 2*y))*p

da = -0.5*dx**2*p
db = -dx*dy*p
dc = -0.5*dy**2*p

print(du == sp.diff(p, u))
print(dv == sp.diff(p, v))



# db1 = sp.diff(p, b)
# db = -(-u + x)*(-v + y)*exp(-0.5*a*(-u + x)**2 - b*(-u + x)*(-v + y) - 0.5*c*(-v + y)**2)




print(da == sp.diff(p, a))
print(sp.simplify(db) == sp.simplify(sp.diff(p, b)))
print(dc == sp.diff(p, c))


