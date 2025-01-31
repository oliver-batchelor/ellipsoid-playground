import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=float, default=2)
    args = parser.parse_args()

    # Domain: -2 to 2
    x = np.linspace(-2, 2, 500)

    # Raised cosine zeros at ±2
    cos_zero = (np.pi / 2) / (2 ** args.k)
    raised_cos = np.clip(
        np.cos((np.abs(x) ** args.k) * cos_zero),
        0, 1
    )
    # "Half" raised cosine zeros at ±2
    half_cos_zero = np.pi / (2 ** args.k)

    half_cos = np.clip(
        0.5 + 0.5 * np.cos((np.abs(x) ** args.k) * half_cos_zero),
        0, 1
    )

    epan = np.clip(1 - (np.abs(x) / 2) ** args.k, 0, 1)

    # Scale for Gaussian so that it is ~0.01 at x=±2
    # => gauss(±2)=0.01 => exp(-((2^k)^2 / alpha^2))=0.01 => alpha=2^k/sqrt(-ln(0.01))
    alpha = 2 ** args.k / np.sqrt(-np.log(0.01))
    gauss = np.exp(-((np.abs(x) ** (2 * args.k)) / alpha ** 2))

    plt.figure(figsize=(8, 5))
    plt.plot(x, raised_cos, label="Raised Cosine f(x)=cos(x^k)")
    plt.plot(x, half_cos, label="Half Raised Cosine f(x)=0.5+0.5*cos(x^k)")
    plt.plot(x, epan, label="Epanechnikov f(x)=1-x^2")
    plt.plot(x, gauss, label="Generalized Gaussian f(x)=exp(-(x^2k))", color="orange")

    plt.title(f"Kernel functions for gaussian (k={args.k}) normalized to 0.01 at ±2")
    plt.xlabel("x")
    plt.ylabel("Function value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    main() 