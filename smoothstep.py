import numpy as np
import matplotlib.pyplot as plt


import sympy as sp

x = sp.symbols('x', real=True)
z = (1 - (sp.Abs(x) / 3))

p = 3*z**2 - 2*z**3

y = sp.integrate(p, x)

def cdf(x):
    return  1.5 + np.sign(x) * x**4/54 - x**3/9 + x


def smoothstep(x):
    z = (1 - (np.abs(x) / 3)).clip(0, 1)
    return 3*z**2 - 2*z**3



def S(x):
    return 1 / (1 + np.exp(-1.6 * x - 0.07 * x**3))

def dS(x):
    sx = S(x)
    return (1.6 + 0.21 * x**2) * sx * (1 - sx)


def S_ab(x, step=0.5):
    return (S(x + step) - S(x - step)) / (2 * step)



def smoothstep4(x):
    z = ((x ** 2) / 3)
    return (1 - z / 3.2).clip(min=0) ** 4


x = np.linspace(-4, 4, 1000)


#show vertical lines at 3 and -3
plt.axvline(3, color='r')
plt.axvline(-3, color='r')


plt.plot(x, np.exp(0.5 * -x**2) , label='exp(-x**2)')
plt.plot(x, smoothstep(x), label='smoothstep(x)')

# plt.plot(x, cdf(x), label='cdf(x)')
# plt.plot(x, dS(x), label='diff_cdf(x)')
# plt.plot(x, S(x), label='cdf_sig(x)')

# plt.plot(x, S_ab(x, 0.2), label='cdf_sig_ab(x)')

plt.plot(x, cdf(x), label='cdf(x)')


# plt.plot(x, smoothstep4(x), label='smoothstep4(x)')

plt.legend()
plt.show()