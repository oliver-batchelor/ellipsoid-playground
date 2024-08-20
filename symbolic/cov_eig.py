import symengine as sp
from symengine import exp

from util import multi_num_equal, num_equal

def parameters():
  mx, my, ux, uy,  s1, s2, vx, vy = sp.symbols('mx my ux uy s1 s2 vx vy')

  u = sp.Matrix([ux, uy]) # pixel centre `u`` in paper
  m = sp.Matrix([mx, my]) # gaussian mean - \hat{u} in paper

  v1 = sp.Matrix([vx, vy]) # direction vector, first eignevector of the covariance matrix

  return u, m, v1, s1, s2


def perp(v):
  return sp.Matrix([-v[1], v[0]])

def eval_intensity(u, m, v1, s1, s2):
  """ Evaluate the intensity and its gradients w.r.t. the parameters
      2D gaussian parameterised by mean, eigenvector and two scales """
  d = u - m

  tx = d.dot(v1)  / s1
  ty = d.dot(perp(v1)) / s2

  tx2, ty2 = tx**2, ty**2

  p = exp(-0.5 * (tx2 + ty2))

  ds1_dp = p * tx2 / s1
  ds2_dp = p * ty2 / s2

  dv1_dp = p * (tx/s1 * -d + ty/s2 * perp(d))
  dm_dp = p * (tx/s1 * v1 + ty/s2 * perp(v1))
  
  return p, dm_dp, ds1_dp, ds2_dp, dv1_dp


u, m, v1, s1, s2 = parameters()
i, dm_dp, ds1_dp, ds2_dp, dv1 = eval_intensity(u, m, v1, s1, s2)


num_equal("ds1_dp", sp.diff(i, s1), ds1_dp)
num_equal("ds2_dp", sp.diff(i, s2), ds2_dp)
multi_num_equal("dv1", (sp.diff(i, v1[0]), sp.diff(i, v1[1])), dv1)

multi_num_equal("dm_dp", (sp.diff(i, m[0]), sp.diff(i, m[1])), dm_dp)






