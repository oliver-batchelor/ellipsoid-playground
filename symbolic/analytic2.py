import symengine as sp
from util import multi_num_equal, num_equal


def perp(v):
    return sp.Matrix([-v[1], v[0]])


def intensity_with_grad(u, m, v1, s1, s2):

  v2 = perp(v1) # second eigenvector

  # local coordinate of pixel centre in gaussian coordinate system
  # \tilde{u} in paper (tx, ty)
  d = u - m

  tx = d.dot(v1)
  ty = d.dot(v2)


  def S(x, sigma=1):
      """ Approximate gaussian cdf """
      z = x / sigma
      return 1 / (1 + sp.exp(-1.6 * z - 0.07 * z**3))


  Sx1, Sx2 = S(tx + 0.5, s1), S(tx - 0.5, s1)
  Sy1, Sy2 = S(ty + 0.5, s2), S(ty - 0.5, s2)
    
  Sx = Sx1 - Sx2
  Sy = Sy1 - Sy2

  i1 = s1 * Sx
  i2 = s2 * Sy

  tau = 2 * sp.pi
  i_2d = tau * i1 * i2


  def dS(sx, x, sigma=1):
      """ Derivative of the approx gaussian cdf S at x """
      return (1.6 + 0.21 * (x/sigma)**2) * sx * (1 - sx)
  

  dSx =  dS(Sx1, tx, s1) - dS(Sx2, tx, s1)
  dSy = dS(Sy1, ty, s2) - dS(Sy2, ty, s2)

                            
  di_dMean = tau * (i2  * dSx * -v1  + i1 * dSy * -v2)

  di_s1 = tau * i2 * ( Sx + dSx * -tx / s1)
  di_s2 = tau * i1 * ( Sy + dSy * -ty / s2)

  di_dv1 = tau * (i2 * dSx * d          # gradient on first eigenvector (v1)
               +  i1 * dSy * -perp(d))  # gradient on second eigenvector (v2 = perp(v1))


  return i_2d, di_dMean, di_s1, di_s2, di_dv1


def parameters():
  mx, my, ux, uy, vx, vy = sp.symbols('mx my ux uy vx vy', real=True)
  s1, s2 = sp.symbols('s1 s2', positive=True, real=True)

  v1 = sp.Matrix([vx, vy]) # direction vector, first eignevector of the covariance matrix
          
  u = sp.Matrix([ux, uy]) # pixel centre `u`` in paper
  m = sp.Matrix([mx, my]) # gaussian mean - \hat{u} in paper

  return u, m, v1, s1, s2

u, m, v1, s1, s2 = parameters()

i_2d, di_dMean, di_s1, di_s2, di_dv1 = intensity_with_grad(u, m, v1, s1, s2)


multi_num_equal("di_dMean", (sp.diff(i_2d, m[0]), sp.diff(i_2d, m[1])), di_dMean)
