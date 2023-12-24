import numpy as np
from bounds import create_obb, radii_from_cov, cov_axes_kernel, OBB, AABB
from taichi.math import mat2, vec2, ivec2

import taichi as ti

ti.init(debug=True)


# @ti.kernel
# def grid_hits(grid:ti.types.ndarray(ti.u1, 2), centre:vec2, dir:vec2, scales:vec2, tile_size:ti.f32):
#   obb = create_obb(centre, dir, scales)

#   for i, j in ti.ndrange(grid.shape[0], grid.shape[1]):
#       tile_min = ti.Vector([i, j]) * tile_size
#       tile_max = tile_min + tile_size
      
#       tile_aabb = AABB(tile_min, tile_max)    
#       # print(obb.corners)
#       # grid[i, j] = not (tile_aabb.separates(obb.corners) or  obb.separates(tile_aabb.corners()))
#       grid[i, j] = not (obb.separates(tile_aabb.corners()))


@ti.kernel
def points_inside(inside:ti.types.ndarray(ti.u1, 1), 
                  points : ti.types.ndarray(vec2, 1),
                  
                  centre:vec2, dir:vec2, scales:vec2):
  obb = create_obb(centre, dir, scales)


  corners = obb.corners
  for i in ti.static(range(4)):
     print(obb.basis @ (corners[i, :] - obb.centre))

  for i in range(points.shape[0]):
      # print(obb.corners)
      # grid[i, j] = not (tile_aabb.separates(obb.corners) or  obb.separates(tile_aabb.corners()))
      inside[i] = obb.inside(points[i])

while True:
  grid_size = 10
  centre = np.random.rand(2) * grid_size + 2
  dir = np.random.randn(2)
  dir = dir / np.linalg.norm(dir)

  scales = np.random.rand(2) * 4 + 0.1

  grid = np.zeros((grid_size + 4, grid_size + 4), dtype=bool)
  # obb = create_obb(centre, dir, scales)
  v1 = dir * scales[0]
  v2 = np.array([-dir[1], dir[0]]) * scales[1]

  corners = np.array([centre + v1 + v2, centre + v1 - v2, centre - v1 - v2, centre - v1 + v2])

  basis = np.linalg.inv(np.array([v1, v2]).T)
  print(v1, v2, np.array([v1, v2]).T)

  grid = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]))
  grid = np.stack(grid, axis=-1).reshape(-1, 2)

  inside = np.zeros((grid.shape[0],), dtype=bool)
  points_inside(inside, grid, centre, dir, scales)


  import matplotlib.pyplot as plt


  plt.scatter(grid[inside, 0], grid[inside, 1], color='r')
  plt.scatter(grid[~inside, 0], grid[~inside, 1], color='b')

  plt.gca().add_patch(plt.Polygon(corners, closed=True, fill=False))


  # # draw grid cells
  # for i, j in np.ndindex(grid.shape):
  #   if grid[i, j]:
  #     plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fill=False))

  plt.show()