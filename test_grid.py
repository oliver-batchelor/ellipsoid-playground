import numpy as np
from bounds import create_obb, radii_from_cov, cov_axes_kernel, OBB, AABB
from taichi.math import mat2, vec2, ivec2

import taichi as ti

ti.init(debug=True)


@ti.kernel
def grid_hits(grid:ti.types.ndarray(ti.u1, 2), centre:vec2, dir:vec2, scales:vec2, tile_size:ti.f32):
  obb = create_obb(centre, dir, scales)

  for i, j in ti.ndrange(grid.shape[0], grid.shape[1]):
      tile_min = ti.Vector([i, j]) * tile_size
      tile_max = tile_min + tile_size
      
      tile_aabb = AABB(tile_min, tile_max)    
      # print(obb.corners)
      # grid[i, j] = not (tile_aabb.separates(obb.corners) or  obb.separates(tile_aabb.corners()))
      grid[i, j] = not (obb.separates(tile_aabb.corners()))


while True:
  grid_size = 5
  centre = np.random.rand(2) * grid_size + 2
  dir = np.random.randn(2)
  dir = dir / np.linalg.norm(dir)

  scales = np.random.rand(2) * 2 + 0.1

  grid = np.zeros((grid_size + 4, grid_size + 4), dtype=bool)
  # obb = create_obb(centre, dir, scales)
  v1 = dir * scales[0]
  v2 = np.array([-dir[1], dir[0]]) * scales[1]

  corners = np.array([centre + v1 + v2, centre + v1 - v2, centre - v1 - v2, centre - v1 + v2])

  basis = np.linalg.inv(np.array([v1, v2]).T)
  print(v1, v2, np.array([v1, v2]).T)

  for c in corners:
    print(basis @ (c - centre))


  grid_hits(grid, centre, dir, scales, 1)


  import matplotlib.pyplot as plt
  import matplotlib.patches as patches

  # set bounds of plot to (10, 10)
  plt.xlim(0, grid.shape[0])
  plt.ylim(0, grid.shape[1])

  # plot oriented box
  
  plt.gca().add_patch(plt.Polygon(corners, closed=True, fill=False))

  # draw grid cells
  for i, j in np.ndindex(grid.shape):
    if grid[i, j]:
      plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fill=False))

  # draw ellipse use matplotlib.patches.Ellipse
  plt.gca().add_patch(patches.Ellipse(centre, scales[0] * 2, scales[1] * 2, angle=np.arctan2(dir[1], dir[0]) * 180 / np.pi, fill=False))    
  
      

  plt.show()