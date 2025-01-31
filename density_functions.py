import numpy as np
import matplotlib.pyplot as plt

def main():
    # Domain: -3 to 3
    x = np.linspace(-2, 2, 500)

    # Unnormalized Gaussian
    z = np.clip(x ** 2, -np.pi, np.pi) 

    gauss = np.exp(- abs(x)**4)

    raised_cos = np.clip(np.cos(z) , 0, 1)
    half_cos = np.clip(0.5 + 0.5 * np.cos(z), 0, 1)

    epan = np.clip(1 - (x/2) **2 , 0, 1)


    # Plot both curves on the same figure
    plt.figure(figsize=(8, 5))
    plt.plot(x, half_cos, label="Half Raised Cosine(k=2)")
    plt.plot(x, raised_cos, label="Raised Cosine(k=2)")
    plt.plot(x, gauss, label="Gaussian", color="orange")
    plt.plot(x, gauss, label="Gen Gaussian(k=4)", color="orange")

    plt.plot(x, epan, label="Epanechnikov")



    plt.xlabel("x")
    plt.ylabel("function value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()