import symengine as sp
from util import multi_num_equal, num_equal


def perp(v):
    return sp.Matrix([-v[1], v[0]])


def intensity_with_grad(u, m, v1, s1, s2):

  v2 = perp(v1) # second eigenvector
  d = u - m # relative position of pixel centre to gaussian mean

  # local coordinate of pixel centre in gaussian coordinate system
  # \tilde{u} in paper (tx, ty)
  tx = d.dot(v1)
  ty = d.dot(v2)

  def S(x, sigma=1):
      """ Approximate gaussian cdf """
      z = x / sigma
      return 1 / (1 + sp.exp(-1.6 * z - 0.07 * z**3))
  
  def dS(sx, x, sigma=1):
      """ Derivative of the approx gaussian cdf S at x """
      return (1.6 + 0.21 * (x/sigma)**2) * sx * (1 - sx)

  tx1, tx2 = tx + 0.5, tx - 0.5
  ty1, ty2 = ty + 0.5, ty - 0.5

  Sx1, Sx2 = S(tx1, s1), S(tx2, s1)
  Sy1, Sy2 = S(ty1, s2), S(ty2, s2)
    
  dSx1, dSx2 =  dS(Sx1, tx1, s1), dS(Sx2, tx2, s1)
  dSy1, dSy2 =  dS(Sy1, ty1, s2), dS(Sy2, ty2, s2)

  # forward pass, computation of intensity
  i1 = s1 * (Sx1 - Sx2)
  i2 = s2 * (Sy1 - Sy2)

  tau = 2 * sp.pi
  i_2d = tau * i1 * i2

  # backward pass, computation of gradients of intensity w.r.t. parameters
  di_dMean = tau * (i2  * (dSx1 - dSx2) * -v1  + i1 * (dSy1 - dSy2) * -v2)

  di_s1 = tau * i2 * ((Sx1 - Sx2) +  (dSx1  * tx1 -  dSx2  * tx2) / -s1)
  di_s2 = tau * i1 * ((Sy1 - Sy2) +  (dSy1  * ty1 -  dSy2  * ty2) / -s2)

  di_dv1 = tau * (i2 * (dSx1 - dSx2) * d          # gradient on first eigenvector (v1)
               +  i1 * (dSy1 - dSy2) * -perp(d))  # gradient on second eigenvector (v2 = perp(v1))


  return i_2d, di_dMean, di_s1, di_s2, di_dv1


def parameters():
  mx, my, ux, uy, vx, vy = sp.symbols('mx my ux uy vx vy', real=True)
  s1, s2 = sp.symbols('s1 s2', positive=True, real=True)

  v1 = sp.Matrix([vx, vy]) # direction vector, first eignevector of the covariance matrix
          
  u = sp.Matrix([ux, uy]) # pixel centre `u`` in paper
  m = sp.Matrix([mx, my]) # gaussian mean - \hat{u} in paper

  return u, m, v1, s1, s2


if __name__ == "__main__":
  u, m, v1, s1, s2 = parameters()
  i_2d, di_dMean, di_s1, di_s2, di_dv1 = intensity_with_grad(u, m, v1, s1, s2)


  multi_num_equal("di_dMean", (sp.diff(i_2d, m[0]), sp.diff(i_2d, m[1])), di_dMean)
  num_equal("di_s1", sp.diff(i_2d, s1), di_s1)
  num_equal("di_s2", sp.diff(i_2d, s2), di_s2)

  multi_num_equal("di_dv1", (sp.diff(i_2d, v1[0]), sp.diff(i_2d, v1[1])), di_dv1)

