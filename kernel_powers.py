import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

def plot_cosine(x, k):
    raised_cos = np.clip(
        np.cos(np.abs(x)**k * np.pi/2),
        0, 1
    )
    return raised_cos

def plot_half_cosine(x, k):
    half_cos = np.clip(
        0.5 + 0.5 * np.cos(np.abs(x)**k * np.pi),
        0, 1
    )
    return half_cos

def plot_gauss(x, k, threshold=0.01):
    return threshold ** (np.abs(x) ** (2 * k)) - 1

def plot_beta(x, k):
    return (1 - x**2) ** (4/k)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, default="cosine", help="Kernel type cosine | half_cosine | beta")
    args = parser.parse_args()

    plt.figure(figsize=(8, 5))

    options = dict(
        cosine=plot_cosine,
        half_cosine=plot_half_cosine,
        beta=plot_beta,
        gauss=plot_gauss
    )


    # Domain: -2 to 2
    x = np.linspace(-1, 1, 2000)
    func = options[args.kernel]
    name = args.kernel

    for i in range(1, 12):
      k = 1.5 ** (i - 4)
      y = func(x, k)
      plt.plot(x, y, label=f"k={k}")

    plt.title(f"Kernel functions for {name}")
    plt.xlabel("x")
    plt.ylabel("Function value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    main() 