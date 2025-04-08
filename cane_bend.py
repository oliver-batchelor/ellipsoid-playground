import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
# Parameters
N = 100
cane_length = 1.0

max_curvature = math.radians(180) # degrees / meter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

wire_height = 0.5  # Height of the wire


# First point fixed, second point at initial angle
initial_angle = torch.pi / 2 * torch.randn(1, device=device)  # Initial random angle around straight-up
dir = torch.tensor([torch.cos(initial_angle), torch.sin(initial_angle), 0.0], device=device)

base_point = torch.randn(1, 3, device=device) * 0.01 + torch.tensor([0.0, wire_height - 0.2, 0.0], device=device)

init_points = torch.linspace(0,  cane_length, N, device=device).view(-1, 1) * dir + base_point


points = torch.nn.Parameter(init_points[2:])  # Don't include the first two fixed points
initial_seg = init_points[:2]

# Optimizer
optimizer = torch.optim.Adam([points], lr=0.01)

def curvatures(points):
  d = F.normalize(points[1:] - points[:-1])  # Differences between consecutive points
  v1, v2 = d[:-1], d[1:]  # Two consecutive segments
  
  cos_theta =  (v1 * v2).sum(dim=1) 
  angles = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

  return angles 


# Loss function
def loss_fn(points):
  # Rebuild full points (include base and second fixed point)
  full_points = torch.cat([initial_seg, points], dim=0)

  print("wire", (full_points[:, 1] - wire_height))
  # Encourage closeness to the wire (at same x)
  follow_loss = torch.mean((full_points[:, 1] - wire_height) ** 2)
  curvature = curvatures(full_points)

  print(max_curvature, curvatures(full_points))
  curvature_loss = ((max_curvature - curvature) / max_curvature).pow(2)
  print(curvature_loss)

  # Attach last point to the wire
  pull_loss = torch.abs(full_points[-1:, 0] - 1.5).mean() 

  lengths = (full_points[1:] - full_points[:-1]).norm(dim=1)

  return (
    
    0.0001 * pull_loss
    + 0.01 * follow_loss 
    + 1.0 * (lengths - cane_length/N).pow(2).mean()
    # +  0.0 * (curvature_loss).mean() 
  )



for i in range(100):
  optimizer.zero_grad()
  loss = loss_fn(points)
  loss.backward()
  optimizer.step()

# Plot
pts = points.detach().cpu().numpy()
# Rebuild full points for plotting (include the base and second fixed points)
full_points = torch.cat([initial_seg, points], dim=0).detach().cpu().numpy()


plt.plot(full_points[1:, 0], full_points[1:, 1], '-o', label='Cane')
plt.axhline(wire_height, color='gray', linestyle='--', label='Wire')
plt.plot(full_points[:2, 0], full_points[:2, 1], '-o', color='red', label='Base')
plt.axis('equal')
plt.title("Cane Fitting to Wire (First Two Segments Fixed)")
plt.legend()
plt.show()
