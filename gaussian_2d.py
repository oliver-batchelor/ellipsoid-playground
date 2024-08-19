import sympy as sp
from sympy import exp, log
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt

# u = Matrix(sp.symbols('ux uy uz'))
# v = Matrix(sp.symbols('vx vy vz'))

# p = Matrix(sp.symbols('px py pz'))

# fx, fy = sp.symbols('fx fy')
# cx, cy = sp.symbols('cx cy')

# K = Matrix(3, 3, [fx, 0, cx, 0, fy, cy, 0, 0, 1])
# P = Matrix(4, 4, [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0, 0, 0, 1, 0])

# # 4x4 homography
# H = Matrix(4, 4, 
#            [u[0], v[0], 0, p[0],
#             u[1], v[1], 0, p[1],
#             u[2], v[2], 0, p[2],
#             0, 0, 0, 1])
            

# M = P @ H

m = sp.symbols('m00 m01 m02 m03 m10 m11 m12 m13 m20 m21 m22 m23 m30 m31 m32 m33')
M = Matrix(4, 4, m)


U = Matrix(4, 2, [*M.row(0), *M.row(3)])

x, y = sp.symbols('x y')

hu = Matrix([-1, 0, 0, x]).T @ M
# hv = Matrix([0, -1, 0, y]).T @ M

print(hu)


hu2 = Matrix([-1, x]).T @ U.T
print(hu2)

# u = (hu.y * hv.w - hu.w * hv.y) / (hu.x * hv.y - hu.y * hv.x)
# v = (hu.w * hv.x - hu.x * hv.w) / (hu.x * hv.y - hu.y * hv.x)

# u = (hu[1] * hv[3] - hu[3] * hv[1]) / (hu[0] * hv[1] - hu[1] * hv[0])
# v = (hu[3] * hv[0] - hu[0] * hv[3]) / (hu[0] * hv[1] - hu[1] * hv[0])

print(u, v)

# p = m @ ti.Vector([u, v, 1., 1.])

# g = ti.exp(-((u**2 + v**2) / 2)**beta)