import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as patches



def dS(x):
    """ Derivative of the approx gaussian cdf S at x """
    sx = S(x)
    return (1.6 + 0.21 * x**2) * sx * (1 - sx)





def show_1d(std=1):
  x = np.linspace(-4 * std, 4 * std, 1000)


  #show vertical lines at 3 and -3
  plt.axvline(3 * std, color='r')
  plt.axvline(-3 * std, color='r')


  # plt.plot(x, dS(x), label='diff_cdf(x)')
  plt.plot(x, S(x, std), label='cdf_sig(x)')

  plt.plot(x, S_pixel(x, std) * np.sqrt(2 * np.pi), label='integral_pixel(x)')


  plt.legend()
  plt.show()


def conic_grid(grid, basis, centre,  blur=0.3):
  cov = basis.T @ basis
  cov[0, 0] += blur
  cov[1, 1] += blur
  inv_cov = np.linalg.inv(cov)

  a, b, c = inv_cov[0, 0], inv_cov[0, 1], inv_cov[1, 1] 
  offsets = (grid - centre)

  x, y = offsets[..., 0], offsets[..., 1]
  d = 0.5 * (a * x ** 2 + c * y ** 2) + b * x * y
  opacity = np.exp(-d)

  return opacity

def perp(v):
  return np.array([-v[1], v[0]])


def S(x, std=1):
    """ Approximate gaussian cdf """
    z = x / std
    return 1 / (1 + np.exp(-1.6 * z - 0.07 * z**3))

def S_pixel(x, std):
    """ Evaluate the integral approximation of the gaussian cdf between x + step and x - step """
    return S(x + 0.5, std) - S(x - 0.5, std) 

def cdf_grid(grid, v, stds, centre):
  sx, sy = stds

  d =  (grid - centre) 
  u = np.dot(d, v)
  v = np.dot(d, perp(v))

  opacity = 2 * np.pi * sx * sy * S_pixel(u, sx) * S_pixel(v, sy)
  return opacity
   

def show_2d():

  gaussian_scale = 2.0

  # generate random basis
  v1 = np.random.randn(2) 
  v1 /= np.linalg.norm(v1)

  scales = np.random.rand(2) * 5 

  v2 = np.array([v1[1], -v1[0]])

  n = 30

  centre = np.array([n // 2, n // 2])
  basis = (np.array([v1, v2]) * scales).T

  
  p = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
  corners = centre.reshape(1, 2) + (p @ basis * gaussian_scale)

  grid = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing='ij'), axis=-1) + 0.5
  
  print(scales)
  opacity_conic = conic_grid(grid, basis, centre, 0.5)
  opacity_cdf = cdf_grid(grid, v1, scales, centre)


  # set bounds of plot to (10, 10)
  plt.xlim(0, grid.shape[0])
  plt.ylim(0, grid.shape[1])

  # plot oriented box
  

  # draw grid cells
  for i, j in np.ndindex(grid.shape[:2]):
     # fill with color gradient based on opacity

    p1 = opacity_cdf[i, j]
    p2 = opacity_conic[i, j]

    plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fill=True, color=(p1, 0, p2, 1)))
    
    
    plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fill=False, color='w'))

      # plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fill=False))

  # plt.gca().add_patch(plt.Polygon(corners, closed=True, fill=False))

  # draw dotted lines along edges extending to infinity
  for i in range(4):
    c1 = corners[i]
    c2 = corners[(i + 1) % 4]

    # draw infinite line which intersects c1 and c2
    slope = (c2[1] - c1[1]) / (c2[0] - c1[0])
    plt.axline(c1, slope=slope, color="w", linestyle=(0, (5, 5)))



  # draw ellipse use matplotlib.patches.Ellipse
  plt.gca().add_patch(patches.Ellipse(centre, 2 * scales[0] * gaussian_scale, 2 * scales[1] * gaussian_scale, 
                                      angle=np.arctan2(v1[1], v1[0]) * 180 / np.pi, fill=False))    
  

  # ensure plot has equal aspect ratio
  plt.gca().set_aspect('equal', adjustable='box')
  
      
  plt.show()

while True:
  show_2d()

# show_1d(4)