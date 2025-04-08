import argparse
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
  return np.log(x) - np.log(1 - x)

# Define the transformed function
def soft_threshold(t, threshold=0.5, margin=4.0):
  return sigmoid((t - threshold) * margin/threshold)

def soft_threshold_gradient(t, threshold=0.5, margin=4.0):
  s = sigmoid((t - threshold) * margin/threshold)
  return margin/threshold * s * (1 - s)

def saturate(t, gain=4.0, k=1.0):
  return (1 - 1/np.exp(gain * t)) ** k

def saturate_gradient(t, gain=4.0, k=1.0):
  return gain * k * (1/np.exp(gain * t)) * (1 - 1/np.exp(gain * t)) ** (k - 1)

args = argparse.ArgumentParser()
args.add_argument("--threshold", type=float, default=0.5)
args.add_argument("--margin", type=float, default=4.0)
args.add_argument("--gain", type=float, default=4.0)
args.add_argument("--k", type=float, default=2.0)
args = args.parse_args()

# Generate values for t from 0 to 1
t_values = np.linspace(0, 1, 500)
# x = soft_threshold(t_values, args.threshold, args.margin)
# grad = soft_threshold_gradient(t_values, args.threshold, args.margin)

s = saturate(t_values, args.gain, args.k)
s_grad = saturate_gradient(t_values, args.gain, args.k)


# Plot the graph
plt.figure(figsize=(8, 5))
# plt.plot(t_values, x, label=r'x', color="blue")
# plt.plot(t_values, grad, label=r'grad', color="red")

plt.plot(t_values, s, label=r's', color="green")
plt.plot(t_values, s_grad, label=r's_grad', color="yellow")

plt.title("Graph of Transformed Sigmoid Function", fontsize=14)
plt.xlabel("t", fontsize=12)
plt.ylabel("sigmoid(t)", fontsize=12)
plt.grid(alpha=0.5)
plt.legend(fontsize=12)
plt.show()
