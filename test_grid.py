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



  
def intersect_circle(scale: float, rows: np.ndarray, centre: np.ndarray):
    
    # Compute x coordinates for valid rows
    local_y = (rows - centre[1]) / scale
    valid = np.abs(local_y) <= 1

    local_y = local_y[valid]
    local_x = np.sqrt(1 - local_y**2) * scale
    local_y *= scale
    
    # Compute intersection points for valid rows
    p1 = centre + np.stack([-local_x, local_y], axis=1)
    p2 = centre + np.stack([local_x, local_y], axis=1)
    
    # Stack points for each valid row
    return np.concatenate([p1, p2], axis=0)


def make_line(normal, point):
    """Create a line equation from a normal vector and point.
    Returns coefficients [a,b,c] for line equation ax + by + c = 0"""
    line = np.append(normal, -np.dot(normal, point))

    return line

def perp(v):
  return np.array([-v[1], v[0]])

def seg_to_line(start, end):
  d = end - start
  d /= np.linalg.norm(d)
  return make_line(perp(d), start)

# def ellipse_lines(centre, dir, scales, rows):
#     import numpy as np

#     # Normalize direction vector
#     dir = dir / np.linalg.norm(dir)
#     ux, uy = dir
#     ux2, uy2 = ux**2, uy**2
#     a_inv_sq = 1 / scales[0]**2
#     b_inv_sq = 1 / scales[1]**2

#     # Ellipse quadratic form coefficients
#     M11 = ux2 * a_inv_sq + uy2 * b_inv_sq
#     M12 = ux * uy * (a_inv_sq - b_inv_sq)
#     M22 = uy2 * a_inv_sq + ux2 * b_inv_sq
#     inv_2A = 1 / (2 * M11)

#     points = []
#     for y in rows:
#         dy = y - centre[1]
#         B = 2 * M12 * dy
#         C = M22 * dy**2 - 1
#         D = B**2 - 4 * M11 * C
#         if D >= 0:
#             sqrt_D = np.sqrt(D)
#             x1 = (-B + sqrt_D) * inv_2A + centre[0]
#             x2 = (-B - sqrt_D) * inv_2A + centre[0]
#             points.extend([[x1, y], [x2, y]])
#     return np.array(points)



def ellipse_aabb(centre:np.ndarray, dir:np.ndarray, scales:np.ndarray):
  v1 = dir * scales[0]
  v2 = np.array([-dir[1], dir[0]]) * scales[1]

  extent = np.sqrt(v1**2 + v2**2)
  return centre - extent, centre + extent


def rand_vec2():
  v = np.random.randn(2)
  return v / np.linalg.norm(v)

while True:
  grid_size = 5
  centre = np.array([5, 5])
  dir = rand_vec2()

  scales = np.random.rand(2) * 2 + 0.2
  
  grid = np.zeros((grid_size + 8, grid_size + 8), dtype=bool)
  # obb = create_obb(centre, dir, scales)
  v1 = dir * scales[0]
  v2 = np.array([-dir[1], dir[0]]) * scales[1]


  corners = np.array([centre + v1 + v2, centre + v1 - v2, centre - v1 - v2, centre - v1 + v2])

  basis = np.linalg.inv(np.array([v1, v2]).T)
  print(v1, v2, np.array([v1, v2]).T)

  for c in corners:
    print(basis @ (c - centre))

  tile_size = 1
  grid_hits(grid, centre, dir, scales, tile_size)


  import matplotlib.pyplot as plt
  import matplotlib.patches as patches

  # set bounds of plot to (10, 10)
  plt.xlim(0, grid.shape[0])
  plt.ylim(0, grid.shape[1])



  # rows = np.arange(grid.shape[1]) * tile_size
  # # print(rows)
  # intersections = ellipse_lines(centre, dir, scales, rows)
  # # # plot as points
  # plt.scatter(intersections[:, 0], intersections[:, 1], c='b', marker='.')


  # show bounds
  bounds_min, bounds_max = ellipse_aabb(centre, dir, scales)
  plt.gca().add_patch(plt.Rectangle(bounds_min, *(bounds_max - bounds_min), fill=False, color='g')) # green

  # draw grid cells
  for i, j in np.ndindex(grid.shape):
    if grid[i, j]:
      plt.gca().add_patch(plt.Rectangle((i, j), 1, 1, fill=False))

  # draw ellipse use matplotlib.patches.Ellipse
  plt.gca().add_patch(patches.Ellipse(centre, scales[0] * 2, scales[1] * 2, angle=np.arctan2(dir[1], dir[0]) * 180 / np.pi, fill=False, color='r'))     #red
  
  # make equal aspect ratio
  plt.gca().set_aspect('equal')

  plt.show()