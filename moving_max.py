import torch


from functools import partial
from matplotlib.widgets import Slider, CheckButtons
import matplotlib.pyplot as plt


def scan(f, xs):
  ys = [xs[0]]
  for i in range(1, len(xs)):
    ys.append(f(xs[i], ys[i-1]))
  return ys

x = torch.randn(1000).pow(2)

def lerp(t, x, y):
  return (1 - t) * x + t * y

def lerp_pow(k):
  def f(t, x, y):
    # Simple power interpolation: x^k * (1-t) + y^k * t
    return ((1 - t) * (x ** k) + t * (y ** k)) ** (1/k)
  return f


def exp_lerp(t, a, b):
    max_ab = torch.maximum(a, b)
    return max_ab + torch.log(lerp(t, torch.exp(a - max_ab), torch.exp(b - max_ab)))

def momentum_max(current, state, beta=0.9):
    """
    Maximum tracking with momentum reserve
    state: (max_est, reserve) tuple
    """
    max_est, reserve = state
    potential_max = max_est + reserve
    
    if current > potential_max:
        # Add to reserve based on how far current is above potential
        new_reserve = reserve + (current - potential_max)
    else:
        # Decay both and add decayed reserve to max_est
        new_reserve = reserve * beta
    max_est = lerp(beta, current, max_est) + reserve * (1 - beta)
    
    return (max_est, new_reserve)



# Modify the figure setup
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(4, 1, height_ratios=[4, 0.2, 0.2, 0.2])
ax_plot = fig.add_subplot(gs[0])
ax_slider1 = fig.add_subplot(gs[1])
ax_slider2 = fig.add_subplot(gs[2])
ax_slider3 = fig.add_subplot(gs[3])

# Adjust margins to reduce space at bottom even more
plt.subplots_adjust(hspace=0.4, bottom=0.08)  # Reduced bottom margin further

# Create separate axes for each checkbox
check_width = 0.15
check_height = 0.04
check_y = 0.01  # Even closer to bottom
labels = ['Input', 'Power', 'Decaying max']
checks = []

for i, label in enumerate(labels):
    ax = plt.axes([0.1 + i*check_width, check_y, check_width, check_height])
    is_active = i <= 1
    check = CheckButtons(ax=ax, labels=[label], actives=[is_active])
    checks.append(check)
    check.on_clicked(partial(lambda _, _i=i: update()))

beta_slider = Slider(
    ax=ax_slider1,
    label='Beta',
    valmin=0.0,
    valmax=1.0,
    valinit=0.9,
)

scale_slider = Slider(
    ax=ax_slider2,
    label='Scale',
    valmin=0.1,
    valmax=10,
    valinit=1.0,
)

k_slider = Slider(
    ax=ax_slider3,
    label='k',
    valmin=1,
    valmax=12,
    valinit=2,
)

def update(*args):
    t = torch.linspace(0, 1, 1000)
    x_scaled = x * scale_slider.val
    beta = beta_slider.val
    k = k_slider.val
    print(f"k value: {k}")

    # Create a new function with k bound
    lerp_func = lerp_pow(k)
    power = scan(lambda a, b: lerp_func(beta, a, b), x_scaled)
    decaying_max = scan(lambda a, b: momentum_max(a, b, beta), x_scaled)
    
    ax_plot.clear()
    
    if checks[0].get_status()[0]:
        ax_plot.plot(t, x_scaled, label='x', linewidth=0.5, alpha=0.5)
    if checks[1].get_status()[0]:
        ax_plot.plot(t, power, label='power')  # Changed label from 'x^k' to 'power'
    if checks[2].get_status()[0]:
        ax_plot.plot(t, decaying_max, label='decaying max')

    ax_plot.legend()
    fig.canvas.draw_idle()

beta_slider.on_changed(update)
scale_slider.on_changed(update)
k_slider.on_changed(update)

update()
plt.show()
