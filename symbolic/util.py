import symengine as sp
from symengine import exp, log, pi, sympify

import numpy as np


def print_cse(exprs):

  cse, derivs = sp.cse(exprs.values(),  optimizations='basic')
  derivs = {k:v for k, v in zip(exprs.keys(), derivs)}

  for k, v in cse:
      print(f"{k} = {v}")

  for k, v in derivs.items():
      print(f"{k} = {v}")


def num_equal(k:str, expr1, expr2, n=100, dps=21):
    
    # Determine over what range to generate random numbers
    sample_min = 1
    sample_max = 10

    # Regroup all free symbols from both expressions
    assert set(expr1.free_symbols) == set(expr2.free_symbols), f"Free symbols do not match: {expr1.free_symbols} != {expr2.free_symbols}"
    free_symbols = list(expr1.free_symbols)
    
    def eval(expr, values):
      subs = {sym: sp.Float(val, dps=dps) for sym, val in zip(free_symbols, values)}
      return float(expr.subs(subs).n(73, real=True))
      
    
    
    # Numeric (brute force) equality testing n-times
    for i in range(n):
        values = np.random.uniform(sample_min, sample_max, len(free_symbols))
        sym_values = list(zip(free_symbols, values))

        n2 = eval(expr2, values)
        n1 = eval(expr1, values)


        if not np.allclose(n1, n2):
            raise AssertionError(f"{k}: Failed for {sym_values}: {n1} != {n2}")          
      
    return True

def multi_num_equal(name, exprs1, exprs2, n=100, dps=21):
    for i, (v1, v2) in enumerate(zip(exprs1, exprs2)):
        num_equal(f"{name}_{i}", v1, v2, n=n, dps=dps)
      
    return True