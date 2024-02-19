import numpy as np
from bounds import radii_from_cov, cov_axes_kernel

import taichi as ti

ti.init(debug=True)

import matplotlib.pyplot as plt
while True:
  radius = np.random.rand()
  dir = np.random.randn(2) 
  dir = dir / np.linalg.norm(dir)

  scales = (np.random.randn(1), 1) 

  def gen_points_with_covariance(dir, scales):
    d1 = dir * scales[0]
    d2 = np.array([-dir[1], dir[0]]) * scales[1]

    points = np.random.randn(1000, 2)
    return points @ np.array([d1, d2]) 


  # plot points 
  points = gen_points_with_covariance(dir, scales)
  
  #compute covariance
  cov = np.cov(points.T)
  print(cov)

  r = radii_from_cov(cov)
  major, minor = cov_axes_kernel(cov)

  print("cov ", cov)

  print("radius", r)
  print("axes", major, minor)

  v1 = dir * scales[0]
  v2 = np.array([-dir[1], dir[0]]) * scales[1]

  print(v1, v2)

  # recompute covariance from axes
  cov1 = np.array([v1, v2]).T @ np.array([v1, v2])
  print("cov2", cov)
  
  plt.gca().set_aspect('equal')

  plt.scatter(points[:, 0], points[:, 1])

  # draw circle with radius r
  circle = plt.Circle((0, 0), r, color='r', fill=False)
  plt.gca().add_artist(circle)

  # draw axis vectors
  plt.plot([0, 3 * major[0]], [0, 3 * major[1]], color='b', linewidth=2)
  plt.plot([0, 3 * minor[0]], [0, 3 * minor[1]], color='c', linewidth=2)

  
  plt.show()
  