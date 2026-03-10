print("Diffusion-limited droplet simulation")
import numpy as np
import matplotlib.pyplot as plt


def droplet_radius(t, r0=1.0, k=0.15):
    """
    Simple diffusion-limited growth law:
        R(t) = (r0^3 + k t)^(1/3)

    Parameters
    ----------
    t : array_like
        Time values.
    r0 : float
        Initial droplet radius.
    k : float
        Growth constant.

    Returns
    -------
    numpy.ndarray
        Droplet radius as a function of time.
    """
    t = np.asarray(t)
    return (r0**3 + k * t) ** (1.0 / 3.0)


def main():
    t = np.linspace(0, 100, 300)
    r = droplet_radius(t)

    plt.figure(figsize=(8, 5))
    plt.plot(t, r, linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Droplet radius")
    plt.title("Diffusion-limited droplet growth")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
