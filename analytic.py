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


def eig_2d(basis):
  x, y, z = basis[0, 0], basis[0, 1], basis[1, 1]
  tr = x + z
  det = x * z - y * y

  gap = tr**2 - 4 * det
  sqrt_gap = np.sqrt(np.maximum(gap, 0))

  lambda1 = (tr + sqrt_gap) * 0.5
  lambda2 = (tr - sqrt_gap) * 0.5

  v1 = np.array([x - lambda2, y])
  v1 /= np.linalg.norm(v1)

  v2 = np.array([-v1[1], v1[0]])

  return (np.sqrt(lambda1), np.sqrt(lambda2)), v1, v2



def eig_grid(grid, centre, v1, scales):
  u = (grid - centre).dot(v1) / scales[0]
  v = (grid - centre).dot(perp(v1)) / scales[1]

  d = (u ** 2 + v ** 2)
  opacity = np.exp(-0.5 * d)

  return opacity



def ellipse_bounds(uv, v1, v2, scales):
  r1, r2 = scales

  extent  = np.sqrt((v1 * r1)**2 +(v2 * r2)**2)
  return (uv - extent), (uv + extent)
  


def cov_grid(grid, cov, centre):
  sigmas, v1, v2 = eig_2d(cov)
  return eig_grid(grid, centre, v1, sigmas)



def draw_grid(opacity, centre, scales, v1, gaussian_scale=2.0):

  # set bounds of plot to (10, 10)
  plt.xlim(0, opacity.shape[0])
  plt.ylim(0, opacity.shape[1])


  # draw grid cells
  for i, j in np.ndindex(opacity.shape[:2]):
     # fill with color gradient based on opacity
    p1 = opacity[i, j]

    plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fill=True, color=(1, 0, 0, p1)))  
    plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fill=False, color='k'))

      # plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fill=False))

  
  # draw dotted lines along edges extending to infinity
  v2 = np.array([v1[1], -v1[0]])
  basis = (np.array([v1, v2]) * scales).T
  
  p = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
  corners = centre.reshape(1, 2) + (p @ basis * gaussian_scale)

  for i in range(4):
    c1 = corners[i]
    c2 = corners[(i + 1) % 4]

    # draw infinite line which intersects c1 and c2
    slope = (c2[1] - c1[1]) / (c2[0] - c1[0])
    plt.axline(c1, slope=slope, color="k", linestyle=(0, (5, 5)), linewidth=1)


  # draw ellipse use matplotlib.patches.Ellipse
  plt.gca().add_patch(patches.Ellipse(centre, 2 * scales[0] * gaussian_scale, 2 * scales[1] * gaussian_scale, 
                                      angle=np.arctan2(v1[1], v1[0]) * 180 / np.pi, fill=False, color='b', linewidth=1.5))    
  
  lower, upper = ellipse_bounds(centre, v1, v2, scales * gaussian_scale)
  # draw box
  plt.gca().add_patch(plt.Rectangle(lower, upper[0] - lower[0], upper[1] - lower[1], fill=False, color='g', linewidth=1.5))

  

  # ensure plot has equal aspect ratio
  plt.gca().set_aspect('equal', adjustable='box')
  
      
  plt.show()

        


def show_2d(cov_blur = 0.3):


  # generate random basis
  v1 = np.random.randn(2) 
  v1 /= np.linalg.norm(v1)

  scales = np.random.rand(2) * 5 

  v2 = np.array([v1[1], -v1[0]])
  basis = (np.array([v1, v2]) * scales).T
  cov = basis.T @ basis

  cov[0, 0] += cov_blur
  cov[1, 1] += cov_blur

  n = 30

  centre = np.array([n // 2, n // 2])

  grid = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing='ij'), axis=-1) + 0.5
    

  print(scales)
  opacity_conic = conic_grid(grid, basis, centre, cov_blur)
  # opacity_cdf = cdf_grid(grid, v1, scales, centre)
  opacity_eig = cov_grid(grid, cov, centre)


  draw_grid(opacity_conic, centre, scales, v1)
  draw_grid(opacity_eig, centre, scales, v1)






while True:
  show_2d()

# show_1d(4)