"""Feynman-Kac agent population simulation with live plots.

Models opinion dynamics + energy under stochastic differential equations.
Agents can die (potential function) and optionally respawn.

Run: python examples/feynman_kac_sim.py
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from chroptiks.plotting_utils import scatter, plothist, makefig

from qstk.feynman_kac import Agent, Environment


def run_simulation():
    """Run a population simulation and plot the results."""

    print("Setting up 50-agent Feynman-Kac simulation...")

    # External difficulty curve: easy start, hard middle, easy end
    external = {
        "difficulty": {0: 0.1, 10: 0.3, 20: 0.9, 30: 0.5, 40: 0.1}
    }

    # Respawn function: occasionally add new agents
    agent_counter = [50]

    def maybe_spawn(env, t):
        if len(env.agents) < 10 and np.random.random() < 0.1:
            agent_counter[0] += 1
            personality = {
                "bias": np.random.uniform(-0.5, 0.5),
                "opinion_adaptability": np.random.uniform(0.02, 0.1),
                "social_susceptibility": np.random.uniform(0.05, 0.2),
                "opinion_volatility": np.random.uniform(0.05, 0.2),
                "energy_recovery_rate": np.random.uniform(0.03, 0.08),
                "energy_volatility": np.random.uniform(0.01, 0.04),
            }
            return Agent(
                agent_counter[0],
                {"opinion": np.random.uniform(-0.5, 0.5), "energy": 0.8},
                personality, env,
                state_bounds={"opinion": (-1, 1), "energy": (0, 1)},
            )
        return None

    env = Environment(external_factors=external, new_agent_fn=maybe_spawn)

    # Create diverse agents
    rng = np.random.default_rng(42)
    for i in range(50):
        personality = {
            "bias": rng.uniform(-0.8, 0.8),
            "opinion_adaptability": rng.uniform(0.02, 0.1),
            "social_susceptibility": rng.uniform(0.05, 0.3),
            "opinion_volatility": rng.uniform(0.05, 0.25),
            "energy_recovery_rate": rng.uniform(0.03, 0.08),
            "energy_volatility": rng.uniform(0.01, 0.04),
        }
        agent = Agent(
            i,
            {"opinion": rng.uniform(-0.5, 0.5), "energy": rng.uniform(0.5, 1.0)},
            personality, env,
            state_bounds={"opinion": (-1, 1), "energy": (0, 1)},
        )
        env.add_agent(agent)

    print("Running simulation (t=0 to t=40, dt=0.05)...")
    history = env.simulate(total_time=40.0, dt=0.05, verbose=True)

    # ── Plotting ────────────────────────────────────────────────────────

    times = [h["time"] for h in history]
    n_agents = [h["num_agents"] for h in history]
    mean_opinion = [h.get("mean_opinion", 0) for h in history]
    var_opinion = [h.get("var_opinion", 0) for h in history]
    mean_energy = [h.get("mean_energy", 0) for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Feynman-Kac Agent Population Simulation", fontsize=16, fontweight="bold")
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # 1. Population over time
    ax = axes[0, 0]
    ax.plot(times, n_agents, "b-", linewidth=2)
    ax.set_title("Population Size")
    ax.set_xlabel("Time")
    ax.set_ylabel("Agents alive")
    ax.grid(True, alpha=0.3)
    # Overlay difficulty
    ax2 = ax.twinx()
    diff_t = np.linspace(0, 40, 200)
    diff_v = [env.get_external_factor("difficulty", t) for t in diff_t]
    ax2.fill_between(diff_t, diff_v, alpha=0.15, color="red", label="Difficulty")
    ax2.set_ylabel("Difficulty", color="red")
    ax2.set_ylim(0, 1.2)

    # 2. Mean opinion over time
    ax = axes[0, 1]
    std_op = [np.sqrt(v) for v in var_opinion]
    ax.plot(times, mean_opinion, "g-", linewidth=2, label="Mean opinion")
    ax.fill_between(times,
                     [m - s for m, s in zip(mean_opinion, std_op)],
                     [m + s for m, s in zip(mean_opinion, std_op)],
                     alpha=0.2, color="green")
    ax.set_title("Opinion Dynamics (mean +/- std)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Opinion")
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 3. Mean energy over time
    ax = axes[0, 2]
    ax.plot(times, mean_energy, "r-", linewidth=2)
    ax.set_title("Mean Energy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 4. Individual agent opinion trajectories (surviving agents)
    ax = axes[1, 0]
    ax.set_title("Individual Opinion Trajectories")
    for agent in env.agents[:20]:  # first 20 survivors
        opinions = [h.get("opinion", 0) for h in agent.history]
        agent_times = np.linspace(0, 40, len(opinions))
        ax.plot(agent_times, opinions, alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Opinion")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)

    # 5. Final opinion distribution (chroptiks histogram)
    ax = axes[1, 1]
    if env.agents:
        final_opinions = [a.state["opinion"] for a in env.agents]
        ax.hist(final_opinions, bins=20, color="#4CAF50", edgecolor="k", alpha=0.8)
    ax.set_title(f"Final Opinion Distribution (n={len(env.agents)})")
    ax.set_xlabel("Opinion")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # 6. Opinion-Energy phase space scatter (chroptiks)
    ax = axes[1, 2]
    if env.agents:
        ops = [a.state["opinion"] for a in env.agents]
        ens = [a.state["energy"] for a in env.agents]
        biases = [a.personality["bias"] for a in env.agents]
        sc = ax.scatter(ops, ens, c=biases, cmap="RdBu_r", s=40, edgecolors="k",
                        linewidths=0.5, vmin=-0.8, vmax=0.8)
        plt.colorbar(sc, ax=ax, label="Bias")
    ax.set_title("Final Opinion-Energy Phase Space")
    ax.set_xlabel("Opinion")
    ax.set_ylabel("Energy")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)

    print(f"\nSimulation complete: {len(env.agents)} agents survived out of {agent_counter[0]} total")
    return fig, env


def plot_survival_analysis(env):
    """Analyze which personality traits predict survival."""

    # We need to track dead agents too — use history length as proxy
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Survival Analysis: What Personality Traits Help?", fontsize=14, fontweight="bold")

    survived = env.agents
    if not survived:
        print("No surviving agents to analyze!")
        return fig

    # Agent lifespan (history length as proxy)
    lifespans = [len(a.history) for a in survived]
    volatilities = [a.personality["opinion_volatility"] for a in survived]
    adaptabilities = [a.personality["opinion_adaptability"] for a in survived]
    susceptibilities = [a.personality["social_susceptibility"] for a in survived]

    ax = axes[0]
    sc = ax.scatter(volatilities, lifespans, c=adaptabilities, cmap="viridis",
                    s=30, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Adaptability")
    ax.set_xlabel("Opinion Volatility")
    ax.set_ylabel("Steps Survived")
    ax.set_title("Volatility vs Survival")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(susceptibilities, lifespans, c=volatilities, cmap="plasma",
                    s=30, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Volatility")
    ax.set_xlabel("Social Susceptibility")
    ax.set_ylabel("Steps Survived")
    ax.set_title("Social Influence vs Survival")
    ax.grid(True, alpha=0.3)

    # Final energy vs bias
    ax = axes[2]
    biases = [a.personality["bias"] for a in survived]
    energies = [a.state["energy"] for a in survived]
    ax.scatter(biases, energies, c=lifespans, cmap="YlOrRd", s=30,
               edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Personality Bias")
    ax.set_ylabel("Final Energy")
    ax.set_title("Bias vs Final Energy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_custom_dynamics():
    """Custom drift/diffusion/potential functions for a different scenario."""

    print("\nRunning custom dynamics: opinion polarization...")

    # Custom: double-well potential drives opinions to +/- 1
    def polarizing_drift(state, personality, env, t):
        op = state.get("opinion", 0)
        # Double-well drift: dV/dx for V(x) = -(x^2)/2 + (x^4)/4
        return {
            "opinion": op - op ** 3 + personality.get("bias", 0) * 0.1,
            "energy": personality.get("energy_recovery_rate", 0.05) - 0.03,
        }

    def noisy_diffusion(state, personality, env, t):
        return {
            "opinion": 0.15 + 0.1 * env.get_external_factor("noise", t),
            "energy": 0.02,
        }

    external = {"noise": {0: 0.1, 10: 0.5, 20: 1.0, 30: 0.3, 40: 0.1}}
    env = Environment(external_factors=external)
    rng = np.random.default_rng(123)

    for i in range(80):
        personality = {
            "bias": rng.uniform(-0.3, 0.3),
            "energy_recovery_rate": rng.uniform(0.03, 0.07),
        }
        agent = Agent(
            i,
            {"opinion": rng.uniform(-0.2, 0.2), "energy": 0.9},
            personality, env,
            drift_fn=polarizing_drift,
            diffusion_fn=noisy_diffusion,
            state_bounds={"opinion": (-1.5, 1.5), "energy": (0, 1)},
        )
        env.add_agent(agent)

    history = env.simulate(total_time=40.0, dt=0.05, verbose=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Custom Dynamics: Double-Well Polarization", fontsize=14, fontweight="bold")

    # Trajectories
    ax = axes[0]
    for agent in env.agents[:30]:
        opinions = [h.get("opinion", 0) for h in agent.history]
        t = np.linspace(0, 40, len(opinions))
        ax.plot(t, opinions, alpha=0.3, linewidth=0.7)
    ax.set_title("Opinion Trajectories (double-well)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Opinion")
    ax.axhline(y=1, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=-1, color="blue", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Opinion histogram evolution
    ax = axes[1]
    times_arr = [h["time"] for h in history]
    for snap_t in [0, 5, 10, 20, 40]:
        idx = min(range(len(times_arr)), key=lambda i: abs(times_arr[i] - snap_t))
        # Get opinions at this time from agent histories
        ops = []
        for agent in env.agents:
            if idx < len(agent.history):
                ops.append(agent.history[idx].get("opinion", 0))
        if ops:
            ax.hist(ops, bins=20, alpha=0.4, label=f"t={snap_t}", density=True,
                    range=(-1.5, 1.5))
    ax.set_title("Opinion Distribution Over Time")
    ax.set_xlabel("Opinion")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Phase portrait: opinion vs d(opinion)/dt
    ax = axes[2]
    x = np.linspace(-1.5, 1.5, 100)
    # Drift field: x - x^3
    drift_field = x - x ** 3
    ax.plot(x, drift_field, "b-", linewidth=2, label="Drift: x - x³")
    ax.fill_between(x, drift_field, alpha=0.1, color="blue")
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.axvline(x=-1, color="red", linestyle=":", alpha=0.7, label="Stable fixed points")
    ax.axvline(x=1, color="red", linestyle=":", alpha=0.7)
    ax.axvline(x=0, color="orange", linestyle=":", alpha=0.7, label="Unstable fixed point")
    ax.set_title("Double-Well Phase Portrait")
    ax.set_xlabel("Opinion (x)")
    ax.set_ylabel("dx/dt")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)
    return fig


if __name__ == "__main__":
    plt.ion()

    print("=" * 60)
    print("Feynman-Kac Agent Simulation Examples")
    print("=" * 60)

    fig1, env = run_simulation()
    fig2 = plot_survival_analysis(env)
    fig3 = plot_custom_dynamics()

    print("\nAll plots open. Close windows or Ctrl+C to exit.")
    plt.show(block=True)
