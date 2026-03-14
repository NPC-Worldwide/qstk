"""Orbital dynamics analysis on synthetic embedding trajectories.

Demonstrates trajectory dynamics, ellipse fitting, orbital classification,
and Lyapunov exponents without needing any LLM calls.

Run: python examples/orbital_dynamics.py
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from chroptiks.plotting_utils import scatter, makefig

from qstk.orbits import (
    compute_trajectory_dynamics,
    compute_orbital_elements,
    fit_ellipse,
    classify_orbit,
    OrbitType,
)


def generate_synthetic_orbit(orbit_type="elliptical", n_points=300, noise=0.05, seed=42):
    """Generate a synthetic 2D orbit trajectory."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_points)

    if orbit_type == "circular":
        r = 2.0
        x = r * np.cos(t) + rng.normal(0, noise, n_points)
        y = r * np.sin(t) + rng.normal(0, noise, n_points)
    elif orbit_type == "elliptical":
        a, b = 3.0, 1.5
        x = a * np.cos(t) + rng.normal(0, noise, n_points)
        y = b * np.sin(t) + rng.normal(0, noise, n_points)
    elif orbit_type == "spiral_in":
        r = 3.0 * np.exp(-0.1 * t)
        x = r * np.cos(t) + rng.normal(0, noise, n_points)
        y = r * np.sin(t) + rng.normal(0, noise, n_points)
    elif orbit_type == "spiral_out":
        r = 0.5 * np.exp(0.08 * t)
        x = r * np.cos(t) + rng.normal(0, noise, n_points)
        y = r * np.sin(t) + rng.normal(0, noise, n_points)
    elif orbit_type == "chaotic":
        # Lorenz-like chaotic orbit projected to 2D
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        x[0], y[0] = 1.0, 0.0
        for i in range(1, n_points):
            dt = 0.02
            dx = 10 * (y[i-1] - x[i-1])
            dy = x[i-1] * (28 - 1) - y[i-1]  # simplified
            x[i] = x[i-1] + dx * dt + rng.normal(0, noise)
            y[i] = y[i-1] + dy * dt + rng.normal(0, noise)
    elif orbit_type == "figure8":
        x = np.sin(t) + rng.normal(0, noise, n_points)
        y = np.sin(2 * t) / 2 + rng.normal(0, noise, n_points)
    else:
        raise ValueError(f"Unknown orbit type: {orbit_type}")

    return np.column_stack([x, y])


def plot_orbit_gallery():
    """Gallery of different orbit types with dynamics analysis."""

    orbit_types = ["circular", "elliptical", "spiral_in", "spiral_out", "chaotic", "figure8"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Orbital Dynamics Gallery", fontsize=16, fontweight="bold")
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    for idx, orbit_type in enumerate(orbit_types):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        positions = generate_synthetic_orbit(orbit_type, n_points=300, noise=0.05)
        dynamics = compute_trajectory_dynamics(positions)

        # Color trajectory by time
        n = len(positions)
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        for i in range(n - 1):
            ax.plot(positions[i:i+2, 0], positions[i:i+2, 1],
                    color=colors[i], linewidth=1.5, alpha=0.7)

        # Start and end markers
        ax.plot(positions[0, 0], positions[0, 1], "go", markersize=10, label="Start", zorder=5)
        ax.plot(positions[-1, 0], positions[-1, 1], "r*", markersize=15, label="End", zorder=5)
        ax.plot(0, 0, "k+", markersize=15, markeredgewidth=2, label="Origin", zorder=5)

        # Fit ellipse and overlay
        if orbit_type not in ["chaotic", "spiral_out"]:
            ellipse_params = fit_ellipse(positions)
            e = Ellipse(
                xy=ellipse_params["center"],
                width=2 * ellipse_params["semi_major"],
                height=2 * ellipse_params["semi_minor"],
                angle=np.degrees(ellipse_params["angle"]),
                fill=False, edgecolor="red", linestyle="--", linewidth=2,
            )
            ax.add_patch(e)

        lyap = dynamics.lyapunov_exponent
        chaotic = "CHAOTIC" if dynamics.is_chaotic else "stable"
        mean_E = float(np.mean(dynamics.total_energy))
        mean_L = float(np.mean(dynamics.angular_momentum))

        ax.set_title(f"{orbit_type}\nλ={lyap:.3f} ({chaotic})  E={mean_E:.2f}  L={mean_L:.2f}",
                     fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_dynamics_detail():
    """Detailed dynamics for a single elliptical orbit."""

    positions = generate_synthetic_orbit("elliptical", n_points=500, noise=0.03, seed=7)
    dynamics = compute_trajectory_dynamics(positions)
    orbital = compute_orbital_elements(positions, dynamics.velocities)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Elliptical Orbit Dynamics — e={orbital.eccentricity:.3f}, "
                 f"type={orbital.orbit_type.value}",
                 fontsize=14, fontweight="bold")
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    t = np.arange(len(positions))

    # 1. Trajectory with velocity vectors
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], "b-", alpha=0.4, linewidth=1)
    step = 20
    ax.quiver(positions[::step, 0], positions[::step, 1],
              dynamics.velocities[::step, 0], dynamics.velocities[::step, 1],
              color="red", scale=2, width=0.004, alpha=0.6)
    ax.plot(0, 0, "k+", markersize=15, markeredgewidth=2)
    ellipse_p = fit_ellipse(positions)
    e = Ellipse(xy=ellipse_p["center"],
                width=2 * ellipse_p["semi_major"], height=2 * ellipse_p["semi_minor"],
                angle=np.degrees(ellipse_p["angle"]),
                fill=False, edgecolor="green", linestyle="--", linewidth=2)
    ax.add_patch(e)
    ax.set_title("Trajectory + Velocity Vectors")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. Speed over time
    ax = axes[0, 1]
    speeds = np.linalg.norm(dynamics.velocities, axis=1)
    ax.plot(t, speeds, color="#2196F3", linewidth=1.5)
    ax.set_title("Speed vs Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed")
    ax.grid(True, alpha=0.3)

    # 3. Energy conservation
    ax = axes[0, 2]
    ax.plot(t, dynamics.kinetic_energy, label="Kinetic", color="#f44336", linewidth=1.5)
    ax.plot(t, dynamics.potential_energy, label="Potential", color="#2196F3", linewidth=1.5)
    ax.plot(t, dynamics.total_energy, label="Total", color="k", linewidth=2)
    ax.set_title("Energy Conservation")
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Angular momentum
    ax = axes[1, 0]
    ax.plot(t, dynamics.angular_momentum, color="#9C27B0", linewidth=1.5)
    ax.set_title("Angular Momentum")
    ax.set_xlabel("Step")
    ax.set_ylabel("L")
    ax.grid(True, alpha=0.3)

    # 5. Distance from origin
    ax = axes[1, 1]
    distances = np.linalg.norm(positions, axis=1)
    ax.plot(t, distances, color="#FF9800", linewidth=1.5)
    ax.axhline(y=orbital.periapsis, color="blue", linestyle="--", alpha=0.7, label=f"Periapsis={orbital.periapsis:.2f}")
    ax.axhline(y=orbital.apoapsis, color="red", linestyle="--", alpha=0.7, label=f"Apoapsis={orbital.apoapsis:.2f}")
    ax.set_title("Distance from Origin")
    ax.set_xlabel("Step")
    ax.set_ylabel("r")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Phase space (r, dr/dt)
    ax = axes[1, 2]
    dr = np.diff(distances)
    sc = ax.scatter(distances[1:], dr, c=t[1:], cmap="viridis", s=5, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Time step")
    ax.set_title("Phase Space (r, ṙ)")
    ax.set_xlabel("r")
    ax.set_ylabel("dr/dt")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)

    # Print orbital elements
    print(f"\n  Orbital Elements:")
    print(f"    Eccentricity:      {orbital.eccentricity:.4f}")
    print(f"    Semi-major axis:   {orbital.semi_major_axis:.4f}")
    print(f"    Periapsis:         {orbital.periapsis:.4f}")
    print(f"    Apoapsis:          {orbital.apoapsis:.4f}")
    print(f"    Specific energy:   {orbital.specific_energy:.4f}")
    print(f"    Angular momentum:  {orbital.angular_momentum:.4f}")
    print(f"    Orbital period:    {orbital.orbital_period:.4f}" if orbital.orbital_period else "    Orbital period:    N/A (unbound)")
    print(f"    Orbit type:        {orbital.orbit_type.value}")
    print(f"    Is bound:          {orbital.is_bound}")
    print(f"    Stability index:   {orbital.stability_index:.4f}")

    return fig


def plot_noise_sensitivity():
    """Show how noise level affects orbit classification."""

    noise_levels = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Noise Sensitivity: Same Ellipse, Increasing Noise",
                 fontsize=14, fontweight="bold")

    results = []
    for idx, noise in enumerate(noise_levels):
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        positions = generate_synthetic_orbit("elliptical", n_points=200, noise=noise, seed=42)
        dynamics = compute_trajectory_dynamics(positions)
        orbital = compute_orbital_elements(positions, dynamics.velocities)
        results.append((noise, orbital))

        ax.plot(positions[:, 0], positions[:, 1], "b.", markersize=2, alpha=0.5)
        if orbital.eccentricity < 2:
            ep = fit_ellipse(positions)
            e = Ellipse(xy=ep["center"],
                        width=2 * ep["semi_major"], height=2 * ep["semi_minor"],
                        angle=np.degrees(ep["angle"]),
                        fill=False, edgecolor="red", linestyle="--", linewidth=2)
            ax.add_patch(e)

        ax.set_title(f"σ={noise}\ne={orbital.eccentricity:.3f} ({orbital.orbit_type.value})\n"
                     f"λ={dynamics.lyapunov_exponent:.3f}", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)

    plt.tight_layout()
    plt.show(block=False)

    # Summary plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    noises = [r[0] for r in results]
    eccs = [r[1].eccentricity for r in results]
    stabs = [r[1].stability_index for r in results]
    ax2.plot(noises, eccs, "o-", color="#f44336", linewidth=2, markersize=8, label="Eccentricity")
    ax2_r = ax2.twinx()
    ax2_r.plot(noises, stabs, "s-", color="#2196F3", linewidth=2, markersize=8, label="Stability")
    ax2.set_xlabel("Noise Level (σ)")
    ax2.set_ylabel("Eccentricity", color="#f44336")
    ax2_r.set_ylabel("Stability Index", color="#2196F3")
    ax2.set_title("Noise vs Orbital Measures")
    ax2.legend(loc="upper left")
    ax2_r.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)

    return fig, fig2


if __name__ == "__main__":
    plt.ion()

    print("=" * 60)
    print("Orbital Dynamics Examples")
    print("=" * 60)

    print("\n[1/3] Orbit type gallery")
    fig1 = plot_orbit_gallery()

    print("\n[2/3] Detailed elliptical orbit analysis")
    fig2 = plot_dynamics_detail()

    print("\n[3/3] Noise sensitivity analysis")
    fig3, fig4 = plot_noise_sensitivity()

    print("\nAll plots open. Close windows or Ctrl+C to exit.")
    plt.show(block=True)
