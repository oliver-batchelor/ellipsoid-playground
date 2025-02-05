import numpy as np

def sample_binomial(n: int, p: float) -> int:
    """
    Sample from a binomial distribution with n trials and probability p
    Args:
        n: number of trials
        p: probability of success for each trial
    Returns:
        number of successes
    """
    # Precompute the ratio terms that don't depend on p
    ratios = [(n-k)/(k+1) for k in range(n)]
    
    u = np.random.rand()
    F = 0.0
    prob = (1 - p)**n
    
    result = n
    for k in range(n):
        F += prob
        if u <= F:
            result = min(k, result)
        prob *= p * ratios[k] / (1.0 - p)
            
    return result




if __name__ == '__main__':
  
  for i in range(10):
      print(sample_binomial(50, 0.8))