import argparse
import numpy as np
import matplotlib.pyplot as plt

def epan_f(x, k):
    return np.clip(1 - (np.abs(x) / 2) ** k, 0, 1)


def compute_kernels(x, args):
    """
    Compute the kernel functions for a given x array and arguments.
    
    Returns a tuple containing:
        - raised_cos: Raised cosine kernel.
        - half_cos: Half raised cosine kernel.
        - epan: Epanechnikov kernel.
        - gauss: Generalized Gaussian kernel (shifted by subtracting the threshold).
    """
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

    epan = epan_f(x, args.k)
    
    threshold = 0.01
    # Note: subtracting 'threshold' does not affect the derivative.
    gauss = threshold ** (np.abs(x * 0.5) ** (2 * args.k)) 

    beta = (1 - (x * 0.5) **2) ** (4/args.k) # * epan_f(x, max(args.k, 2))

    return raised_cos, half_cos, epan, gauss, beta

def plot_functions(x, args):
    # Domain: -2 to 2
    x = np.linspace(-2, 2, 500)
    raised_cos, half_cos, epan, gauss, beta = compute_kernels(x, args)

    plt.figure(figsize=(8, 5))
    plt.plot(x, raised_cos, label="Raised Cosine f(x)=cos(x^k)")
    plt.plot(x, half_cos, label="Half Raised Cosine f(x)=0.5+0.5*cos(x^k)")
    plt.plot(x, epan, label="Epanechnikov f(x)=1-x^k")
    plt.plot(x, gauss, label="Generalized Gaussian f(x)=exp(-(x^2k))", color="orange")
    plt.plot(x, beta, label="Beta f(x)=(1-x^2)^(4/k)", color="purple")

def plot_derivatives(x, args):
    # Domain: -2 to 2 with a finer resolution for gradient computation.
    x = np.linspace(-2, 2, 2000)
    raised_cos, half_cos, epan, gauss, beta = compute_kernels(x, args)

    # Compute gradients using np.gradient
    raised_cos_deriv = np.gradient(raised_cos, x)
    half_cos_deriv = np.gradient(half_cos, x)
    epan_deriv = np.gradient(epan, x)
    gauss_deriv = np.gradient(gauss, x)
    beta_deriv = np.gradient(beta, x)
    plt.figure(figsize=(8, 5))
    plt.plot(x, raised_cos_deriv, label="Derivative Raised Cosine")
    plt.plot(x, half_cos_deriv, label="Derivative Half Raised Cosine")
    plt.plot(x, epan_deriv, label="Derivative Epanechnikov")
    plt.plot(x, gauss_deriv, label="Derivative Generalized Gaussian", color="orange")
    plt.plot(x, beta_deriv, label="Derivative Beta", color="purple")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=float, default=2)
    args = parser.parse_args()

    # Plot kernel functions
    plot_functions(None, args)
    plt.title(f"Kernel functions for gaussian (k={args.k}) normalized to 0.01 at ±2")
    plt.xlabel("x")
    plt.ylabel("Function value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Plot derivatives of kernel functions
    plot_derivatives(None, args)
    plt.title(f"Derivatives of kernel functions (k={args.k})")
    plt.xlabel("x")
    plt.ylabel("Derivative")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main() 