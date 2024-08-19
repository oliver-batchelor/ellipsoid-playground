import sympy as sp
from sympy import exp, log, pi, sympify
import numpy as np
mx, my, ux, uy, vx, vy = sp.symbols('mx my ux uy vx vy')

# s1, s2 = sp.symbols('s1 s2', positive=True)
s1, s2 = sp.symbols('s1 s2', positive=True)

v1 = sp.Matrix([vx, vy]) # direction vector, first eignevector of the covariance matrix
v2 = sp.Matrix([-vy, vx]) # second eigenvector
          
u = sp.Matrix([ux, uy]) # pixel centre `u`` in paper
m = sp.Matrix([mx, my]) # gaussian mean - \hat{u} in paper


# local coordinate of pixel centre in gaussian coordinate system
# \tilde{u} in paper (tx, ty)
d = u - m

tx = d.dot(v1)
ty = d.dot(v2)

def S(x, sigma=1):
    """ Approximate gaussian cdf """
    z = x / sigma
    return 1 / (1 + exp(-sympify('1.6') * z - sympify('0.07') * z**3))

def S_pixel(x, sigma):
    """ Evaluate the integral approximation of the gaussian cdf between x + step and x - step """
    return S(x + sp.sympify('0.5'), sigma) - S(x - sp.sympify('0.5'), sigma) 

i1 = s1 * S_pixel(tx, s1)
i2 = s2 * S_pixel(ty, s2)
p = 2 * sp.pi * i1 * i2



def print_cse(exprs):

  cse, derivs = sp.cse(exprs.values(),  optimizations='basic')
  derivs = {k:v for k, v in zip(exprs.keys(), derivs)}

  for k, v in cse:
      print(f"{k} = {v}")

  for k, v in derivs.items():
      print(f"{k} = {v}")

def test_equal(expr1, expr2):
  expr1 = sp.nsimplify(expr1, rational=True)
  expr2 = sp.nsimplify(expr2, rational=True)

  return sp.simplify(expr1 - expr2) == 0 

def num_equal(expr1, expr2, n=10):
    
    # Determine over what range to generate random numbers
    sample_min = 1
    sample_max = 10

    # Regroup all free symbols from both expressions
    assert set(expr1.free_symbols) == set(expr2.free_symbols), f"Free symbols do not match: {expr1.free_symbols} != {expr2.free_symbols}"
    free_symbols = list(expr1.free_symbols)
    
    def eval(expr, values):
      expr=expr.subs(dict(zip(free_symbols, values)))
      return float(expr)
    
    # Numeric (brute force) equality testing n-times
    for i in range(n):
        values = np.random.uniform(sample_min, sample_max, len(free_symbols))
        n1 = eval(expr1, values)
        n2 = eval(expr2, values)

        if not np.allclose(n1, n2):
            print(f"Failed for {list(zip(free_symbols, values))}: {n1} != {n2}")
            return False

        
    return True


def dS(x, sigma=1):
    """ Derivative of the approx gaussian cdf S at x """
    sx = S(x, sigma)
    return (sympify('1.6') + sympify('0.21') * (x/sigma)**2) * sx * (1 - sx)


def dS_dx(x, sigma=1):
    return dS(x, sigma) * 1/sigma

# def dS_dsigma(x, sigma=1):
#   return dS(x, sigma) * -x/(sigma ** 2)

                                
x = sp.symbols('x')




print(test_equal(sp.diff(S(x, s1), x), dS_dx(x, s1)))

dp_dMean = 2 * pi * (i2  * ((dS_dx(tx + 0.5, s1) - dS_dx(tx - 0.5, s1)) * -v1)
                    + i1 * ((dS_dx(ty + 0.5, s2) - dS_dx(ty - 0.5, s2)) * -v2))



di_dMean = (dS_dx(tx + 0.5, s1) - dS_dx(tx - 0.5, s1)) * -v1

num_equal(sp.diff(S_pixel(x, s1), x), 
          dS_dx(x + 0.5, s1) - dS_dx(x - 0.5, s1), n=100)




# print(sp.diff(p, mx))
# print()
# print(sp.sympify(dp_dMean[0]))

# print(num_equal(sp.diff(p, mx), dp_dMean[0]))




# for k, deriv in derivs.items():
#     print(k, num_equal(expanded[k], derived[k]))
