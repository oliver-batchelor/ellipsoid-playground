import sympy as sp


fx, fy, cx, cy, x, y, z = sp.symbols('fx fy cx cy x y z')

K = sp.Matrix([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])


p = sp.Matrix([x, y, z])

# project point
p_proj_ = K @ p

d = p_proj_[2]
p_proj = p_proj_ / p_proj_[2]


print(d)


# compute jacobian 3x3
J = sp.Matrix([[sp.diff(p_proj[0], x), sp.diff(p_proj[0], y), sp.diff(p_proj[0], z)],
              [sp.diff(p_proj[1], x), sp.diff(p_proj[1], y), sp.diff(p_proj[1], z)]])

print(sp.simplify(J))

J1 = sp.diff(p_proj, p)

J1_flat = sp.transpose(J1.reshape(3, 3))

print(sp.simplify(J1_flat))