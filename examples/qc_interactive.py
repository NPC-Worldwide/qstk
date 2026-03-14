"""Interactive quantum circuit explorer with chroptiks plots.

Run: python examples/qc_interactive.py

Uses matplotlib sliders + chroptiks for live visualization of:
  1. CHSH S-value vs Bob's measurement angle
  2. Werner state noise sweep (S-value + entanglement measures)
  3. Parameterized entanglement (theta controls everything)
  4. Semantic state amplitudes → CHSH violation landscape
  5. Full 2D angle heatmap of S-value
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from chroptiks.plotting_utils import scatter, plot2dhist, plothist, plotbar, makefig

from qstk.qc import (
    bell_state, werner_state, parameterized_entangled_state,
    alice_operators, bob_operators,
    chsh_s_value, chsh_expectation_values, measure_state,
    entanglement_entropy, concurrence, fidelity,
    chsh_circuit, semantic_circuit,
)


# ═══════════════════════════════════════════════════════════════════════
# Plot 1: Interactive CHSH angle explorer
# ═══════════════════════════════════════════════════════════════════════

def plot_chsh_angle_explorer():
    """Drag sliders to change Alice/Bob measurement angles, see S-value update."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CHSH Angle Explorer", fontsize=16, fontweight="bold")
    plt.subplots_adjust(bottom=0.30, hspace=0.35, wspace=0.3)

    # Precompute angle sweep data
    angles = np.linspace(0, 2 * np.pi, 360)
    state = bell_state("phi_plus")

    # --- Axes setup ---
    ax_s = axes[0, 0]
    ax_ev = axes[0, 1]
    ax_bar = axes[1, 0]
    ax_counts = axes[1, 1]

    # Initial angles
    init_a0, init_a1 = 0.0, np.pi / 2
    init_b0, init_b1 = np.pi / 4, 3 * np.pi / 4

    # S-value vs b1 sweep curve
    def compute_sweep(a0, a1, b0):
        ss = []
        for b1 in angles:
            A0, A1 = alice_operators(a0, a1)
            B0, B1 = bob_operators(b0, b1)
            ss.append(chsh_s_value(state, A0, A1, B0, B1))
        return np.array(ss)

    sweep_data = compute_sweep(init_a0, init_a1, init_b0)

    # Top-left: S vs b1 angle
    ax_s.set_title("S-value vs Bob's b₁ angle")
    ax_s.set_xlabel("b₁ (degrees)")
    ax_s.set_ylabel("S")
    line_sweep, = ax_s.plot(np.degrees(angles), sweep_data, "b-", linewidth=2)
    ax_s.axhline(y=2, color="red", linestyle="--", alpha=0.7, label="Classical bound")
    ax_s.axhline(y=-2, color="red", linestyle="--", alpha=0.7)
    ax_s.axhline(y=2 * np.sqrt(2), color="green", linestyle=":", alpha=0.7, label="Tsirelson bound")
    ax_s.axhline(y=-2 * np.sqrt(2), color="green", linestyle=":", alpha=0.7)
    marker_b1, = ax_s.plot(np.degrees(init_b1), 0, "ro", markersize=12, zorder=5)
    ax_s.set_ylim(-3.2, 3.2)
    ax_s.legend(fontsize=8)
    ax_s.grid(True, alpha=0.3)

    # Top-right: 4 expectation values
    ax_ev.set_title("Expectation Values E(Aᵢ, Bⱼ)")
    ax_ev.set_ylabel("E")
    ax_ev.set_ylim(-1.2, 1.2)
    ax_ev.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ev_labels = ["E(A₀,B₀)", "E(A₀,B₁)", "E(A₁,B₀)", "E(A₁,B₁)"]
    ev_colors = ["#2196F3", "#f44336", "#4CAF50", "#FF9800"]
    bars_ev = ax_ev.bar(ev_labels, [0, 0, 0, 0], color=ev_colors, edgecolor="k", linewidth=0.5)
    ax_ev.grid(True, alpha=0.3, axis="y")

    # Bottom-left: S-value gauge
    ax_bar.set_title("S-value")
    ax_bar.set_xlim(-3.5, 3.5)
    ax_bar.set_ylim(0, 1)
    ax_bar.axvline(x=2, color="red", linestyle="--", linewidth=2, label="|S|=2")
    ax_bar.axvline(x=-2, color="red", linestyle="--", linewidth=2)
    ax_bar.axvline(x=2 * np.sqrt(2), color="green", linestyle=":", linewidth=2, label="2√2")
    ax_bar.axvline(x=-2 * np.sqrt(2), color="green", linestyle=":", linewidth=2)
    s_bar = ax_bar.barh(0.5, 0, height=0.6, color="blue", edgecolor="k")
    s_text = ax_bar.text(0, 0.5, "", ha="center", va="center", fontsize=20, fontweight="bold")
    ax_bar.set_yticks([])
    ax_bar.legend(fontsize=8, loc="lower right")
    ax_bar.grid(True, alpha=0.3, axis="x")

    # Bottom-right: measurement histogram
    ax_counts.set_title("Simulated Measurements (1000 shots)")
    ax_counts.set_ylabel("Counts")
    count_bars = ax_counts.bar(["00", "01", "10", "11"], [0, 0, 0, 0],
                                color=["#2196F3", "#f44336", "#4CAF50", "#FF9800"],
                                edgecolor="k", linewidth=0.5)
    ax_counts.set_ylim(0, 1100)
    ax_counts.grid(True, alpha=0.3, axis="y")

    # Sliders
    slider_ax = [
        plt.axes([0.15, 0.18, 0.7, 0.025]),
        plt.axes([0.15, 0.14, 0.7, 0.025]),
        plt.axes([0.15, 0.10, 0.7, 0.025]),
        plt.axes([0.15, 0.06, 0.7, 0.025]),
    ]
    s_a0 = Slider(slider_ax[0], "a₀ (deg)", 0, 360, valinit=np.degrees(init_a0), color="#2196F3")
    s_a1 = Slider(slider_ax[1], "a₁ (deg)", 0, 360, valinit=np.degrees(init_a1), color="#4CAF50")
    s_b0 = Slider(slider_ax[2], "b₀ (deg)", 0, 360, valinit=np.degrees(init_b0), color="#FF9800")
    s_b1 = Slider(slider_ax[3], "b₁ (deg)", 0, 360, valinit=np.degrees(init_b1), color="#f44336")

    def update(val):
        a0 = np.radians(s_a0.val)
        a1 = np.radians(s_a1.val)
        b0 = np.radians(s_b0.val)
        b1 = np.radians(s_b1.val)

        A0, A1 = alice_operators(a0, a1)
        B0, B1 = bob_operators(b0, b1)

        # Expectation values
        ev = chsh_expectation_values(state, A0, A1, B0, B1)
        ev_vals = [ev["A_B"], ev["A_B_prime"], ev["A_prime_B"], ev["A_prime_B_prime"]]
        s_val = ev_vals[0] - ev_vals[1] + ev_vals[2] + ev_vals[3]

        # Update sweep curve
        sweep = compute_sweep(a0, a1, b0)
        line_sweep.set_ydata(sweep)
        marker_b1.set_xdata([np.degrees(b1)])
        marker_b1.set_ydata([s_val])

        # Update EV bars
        for bar_rect, val_i in zip(bars_ev, ev_vals):
            bar_rect.set_height(val_i)

        # Update S gauge
        s_bar[0].set_width(s_val)
        if s_val < 0:
            s_bar[0].set_x(s_val)
        else:
            s_bar[0].set_x(0)
        color = "green" if abs(s_val) > 2 else "gray"
        s_bar[0].set_color(color)
        s_text.set_x(s_val / 2)
        s_text.set_text(f"S={s_val:.3f}")

        # Update measurement counts
        counts = measure_state(state, n_shots=1000, seed=42)
        for bar_rect, label in zip(count_bars, ["00", "01", "10", "11"]):
            bar_rect.set_height(counts.get(label, 0))

        fig.canvas.draw_idle()

    s_a0.on_changed(update)
    s_a1.on_changed(update)
    s_b0.on_changed(update)
    s_b1.on_changed(update)
    update(None)  # initial draw

    plt.show(block=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Plot 2: Werner noise explorer
# ═══════════════════════════════════════════════════════════════════════

def plot_werner_explorer():
    """Slider controls Werner p, see how noise kills entanglement."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Werner State: Noise vs Entanglement", fontsize=16, fontweight="bold")
    plt.subplots_adjust(bottom=0.25, wspace=0.35)

    # Precompute full sweep
    ps = np.linspace(0, 1, 200)
    s_vals = []
    for p in ps:
        rho = werner_state(bell_state("phi_plus"), p)
        s_vals.append(chsh_s_value(rho, is_density_matrix=True))
    s_vals = np.array(s_vals)

    # Left: S vs p curve
    ax_s = axes[0]
    ax_s.set_title("S-value vs Werner p")
    ax_s.set_xlabel("p (entanglement strength)")
    ax_s.set_ylabel("S")
    ax_s.plot(ps, s_vals, "b-", linewidth=2)
    ax_s.axhline(y=2, color="red", linestyle="--", label="Classical bound")
    ax_s.axhline(y=2 * np.sqrt(2), color="green", linestyle=":", label="Tsirelson")
    ax_s.axvline(x=1 / np.sqrt(2), color="orange", linestyle="-.", alpha=0.7, label=f"p=1/√2≈{1/np.sqrt(2):.3f}")
    marker_p, = ax_s.plot(1.0, 2 * np.sqrt(2), "ro", markersize=12, zorder=5)
    ax_s.legend(fontsize=8)
    ax_s.grid(True, alpha=0.3)
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(-0.2, 3.2)

    # Middle: density matrix heatmap
    ax_rho = axes[1]
    ax_rho.set_title("Density Matrix ρ(p)")
    rho_init = werner_state(bell_state("phi_plus"), 1.0)
    im = ax_rho.imshow(np.real(rho_init), cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                        extent=[0, 4, 4, 0])
    ax_rho.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax_rho.set_xticklabels(["00", "01", "10", "11"])
    ax_rho.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax_rho.set_yticklabels(["00", "01", "10", "11"])
    plt.colorbar(im, ax=ax_rho, shrink=0.8)

    # Right: EV bars
    ax_ev = axes[2]
    ax_ev.set_title("CHSH Expectation Values")
    ax_ev.set_ylim(-1.2, 1.2)
    ev_labels = ["E₀₀", "E₀₁", "E₁₀", "E₁₁"]
    ev_colors = ["#2196F3", "#f44336", "#4CAF50", "#FF9800"]
    bars_ev = ax_ev.bar(ev_labels, [0, 0, 0, 0], color=ev_colors, edgecolor="k")
    s_text = ax_ev.text(0.5, 1.05, "", transform=ax_ev.transAxes, ha="center",
                        fontsize=14, fontweight="bold")
    ax_ev.axhline(y=0, color="gray", alpha=0.3)
    ax_ev.grid(True, alpha=0.3, axis="y")

    slider_ax = plt.axes([0.15, 0.08, 0.7, 0.04])
    s_p = Slider(slider_ax, "Werner p", 0.0, 1.0, valinit=1.0, color="#FF9800")

    def update(val):
        p = s_p.val
        psi = bell_state("phi_plus")
        rho = werner_state(psi, p)
        s = chsh_s_value(rho, is_density_matrix=True)

        marker_p.set_xdata([p])
        marker_p.set_ydata([s])

        im.set_data(np.real(rho))

        A0, A1 = alice_operators()
        B0, B1 = bob_operators()
        from qstk.qc import chsh_expectation_values_density
        ev = chsh_expectation_values_density(rho, A0, A1, B0, B1)
        ev_vals = [ev["A_B"], ev["A_B_prime"], ev["A_prime_B"], ev["A_prime_B_prime"]]
        for bar_rect, v in zip(bars_ev, ev_vals):
            bar_rect.set_height(v)

        violation = "VIOLATION" if abs(s) > 2 else "classical"
        color = "green" if abs(s) > 2 else "red"
        s_text.set_text(f"S = {s:.4f}  ({violation})")
        s_text.set_color(color)

        fig.canvas.draw_idle()

    s_p.on_changed(update)
    update(None)
    plt.show(block=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Plot 3: Entanglement parameter explorer
# ═══════════════════════════════════════════════════════════════════════

def plot_entanglement_explorer():
    """Theta/phi sliders control parameterized state, see all measures update."""

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Parameterized Entanglement Explorer", fontsize=16, fontweight="bold")
    plt.subplots_adjust(bottom=0.22, hspace=0.35, wspace=0.3)

    # Precompute theta sweep
    thetas = np.linspace(0, np.pi / 2, 200)
    c_sweep = [concurrence(parameterized_entangled_state(t)) for t in thetas]
    s_sweep = [chsh_s_value(parameterized_entangled_state(t)) for t in thetas]
    e_sweep = [entanglement_entropy(parameterized_entangled_state(t)) for t in thetas]

    # Top-left: concurrence + S-value vs theta
    ax_cs = axes[0, 0]
    ax_cs.set_title("Concurrence & S-value vs θ")
    ax_cs.set_xlabel("θ (degrees)")
    l1, = ax_cs.plot(np.degrees(thetas), c_sweep, "b-", linewidth=2, label="Concurrence")
    ax_cs2 = ax_cs.twinx()
    l2, = ax_cs2.plot(np.degrees(thetas), s_sweep, "r-", linewidth=2, label="S-value")
    ax_cs2.axhline(y=2, color="red", linestyle="--", alpha=0.5)
    ax_cs.set_ylabel("Concurrence", color="blue")
    ax_cs2.set_ylabel("S-value", color="red")
    marker_c, = ax_cs.plot(45, 1, "bo", markersize=10, zorder=5)
    marker_s, = ax_cs2.plot(45, 2 * np.sqrt(2), "rs", markersize=10, zorder=5)
    ax_cs.legend([l1, l2], ["Concurrence", "S-value"], loc="lower center", fontsize=8)
    ax_cs.grid(True, alpha=0.3)

    # Top-right: state vector amplitudes
    ax_amp = axes[0, 1]
    ax_amp.set_title("State Amplitudes |ψ⟩")
    ax_amp.set_ylim(0, 1.1)
    amp_bars = ax_amp.bar(["α|00⟩", "α|01⟩", "α|10⟩", "α|11⟩"],
                           [1, 0, 0, 0],
                           color=["#2196F3", "#ccc", "#ccc", "#f44336"],
                           edgecolor="k")
    ax_amp.grid(True, alpha=0.3, axis="y")

    # Bottom-left: Bloch-like visualization (probabilities pie)
    ax_prob = axes[1, 0]
    ax_prob.set_title("Measurement Probabilities")

    # Bottom-right: info text
    ax_info = axes[1, 1]
    ax_info.set_title("Quantum Measures")
    ax_info.axis("off")
    info_text = ax_info.text(0.1, 0.5, "", fontsize=13, family="monospace",
                             verticalalignment="center", transform=ax_info.transAxes)

    # Sliders
    slider_theta_ax = plt.axes([0.15, 0.10, 0.7, 0.03])
    slider_phi_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    s_theta = Slider(slider_theta_ax, "θ (deg)", 0, 90, valinit=45, color="#2196F3")
    s_phi = Slider(slider_phi_ax, "φ (deg)", 0, 360, valinit=0, color="#f44336")

    def update(val):
        theta = np.radians(s_theta.val)
        phi = np.radians(s_phi.val)
        psi = parameterized_entangled_state(theta, phi)

        c = concurrence(psi)
        s = chsh_s_value(psi)
        ent = entanglement_entropy(psi)
        f = fidelity(psi, bell_state("phi_plus"))

        # Update sweep markers
        marker_c.set_xdata([s_theta.val])
        marker_c.set_ydata([c])
        marker_s.set_xdata([s_theta.val])
        marker_s.set_ydata([s])

        # Update amplitude bars
        amps = np.abs(psi)
        for bar_rect, a in zip(amp_bars, amps):
            bar_rect.set_height(a)

        # Update probabilities pie
        ax_prob.clear()
        ax_prob.set_title("Measurement Probabilities")
        probs = np.abs(psi) ** 2
        nonzero = probs > 0.001
        labels_pie = np.array(["00", "01", "10", "11"])
        colors_pie = ["#2196F3", "#FF9800", "#4CAF50", "#f44336"]
        if np.any(nonzero):
            ax_prob.pie(probs[nonzero], labels=labels_pie[nonzero],
                       colors=np.array(colors_pie)[nonzero],
                       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})

        # Update info
        violation = "YES" if abs(s) > 2 else "no"
        info_text.set_text(
            f"State: cos({s_theta.val:.0f}°)|00⟩ + e^(i{s_phi.val:.0f}°)sin({s_theta.val:.0f}°)|11⟩\n\n"
            f"Concurrence:    {c:.4f}\n"
            f"S-value:        {s:.4f}\n"
            f"CHSH violation: {violation}\n"
            f"Ent. entropy:   {ent:.4f} bits\n"
            f"Fidelity to Φ+: {f:.4f}"
        )

        fig.canvas.draw_idle()

    s_theta.on_changed(update)
    s_phi.on_changed(update)
    update(None)
    plt.show(block=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Plot 4: Semantic state explorer
# ═══════════════════════════════════════════════════════════════════════

def plot_semantic_explorer():
    """4 amplitude sliders for word-sense state, see CHSH violation live."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Semantic CHSH: "bank" × "bat" Word Sense Entanglement',
                 fontsize=14, fontweight="bold")
    plt.subplots_adjust(bottom=0.28, hspace=0.35, wspace=0.3)

    ax_bar = axes[0, 0]
    ax_ev = axes[0, 1]
    ax_pie = axes[1, 0]
    ax_info = axes[1, 1]

    # State amplitudes bar
    ax_bar.set_title("State Amplitudes")
    ax_bar.set_ylim(0, 1.1)
    basis_labels = ["|fin,bat⟩", "|fin,ani⟩", "|riv,bat⟩", "|riv,ani⟩"]
    amp_colors = ["#2196F3", "#FF9800", "#4CAF50", "#f44336"]
    amp_bars = ax_bar.bar(basis_labels, [0.7, 0.05, 0.05, 0.7], color=amp_colors, edgecolor="k")
    ax_bar.grid(True, alpha=0.3, axis="y")

    # EV bars
    ax_ev.set_title("CHSH Expectation Values")
    ax_ev.set_ylim(-1.2, 1.2)
    ev_labels = ["E₀₀", "E₀₁", "E₁₀", "E₁₁"]
    ev_colors_list = ["#2196F3", "#f44336", "#4CAF50", "#FF9800"]
    bars_ev = ax_ev.bar(ev_labels, [0, 0, 0, 0], color=ev_colors_list, edgecolor="k")
    ax_ev.axhline(y=0, color="gray", alpha=0.3)
    ax_ev.grid(True, alpha=0.3, axis="y")

    # Measurement pie
    ax_pie.set_title("Measurement Outcomes")

    # Info panel
    ax_info.axis("off")
    ax_info.set_title("Results")
    info_text = ax_info.text(0.1, 0.5, "", fontsize=13, family="monospace",
                             verticalalignment="center", transform=ax_info.transAxes)

    # 4 amplitude sliders
    sl_axes = [
        plt.axes([0.15, 0.19, 0.3, 0.025]),
        plt.axes([0.55, 0.19, 0.3, 0.025]),
        plt.axes([0.15, 0.14, 0.3, 0.025]),
        plt.axes([0.55, 0.14, 0.3, 0.025]),
    ]
    sl_labels = ["α(fin,bat)", "α(fin,ani)", "α(riv,bat)", "α(riv,ani)"]
    sl_inits = [0.7, 0.05, 0.05, 0.7]
    sliders = []
    for ax_sl, label, init, col in zip(sl_axes, sl_labels, sl_inits, amp_colors):
        s = Slider(ax_sl, label, 0.0, 1.0, valinit=init, color=col)
        sliders.append(s)

    # Presets buttons
    preset_ax = plt.axes([0.15, 0.04, 0.7, 0.06])
    preset_ax.axis("off")
    btn_axes = [
        plt.axes([0.15, 0.04, 0.15, 0.04]),
        plt.axes([0.33, 0.04, 0.15, 0.04]),
        plt.axes([0.51, 0.04, 0.15, 0.04]),
        plt.axes([0.69, 0.04, 0.15, 0.04]),
    ]
    btn_corr = Button(btn_axes[0], "Correlated", color="#e3f2fd")
    btn_anti = Button(btn_axes[1], "Anti-corr", color="#fce4ec")
    btn_uniform = Button(btn_axes[2], "Uniform", color="#e8f5e9")
    btn_product = Button(btn_axes[3], "Product", color="#fff3e0")

    def set_sliders(vals):
        for s, v in zip(sliders, vals):
            s.set_val(v)

    btn_corr.on_clicked(lambda _: set_sliders([0.7, 0.05, 0.05, 0.7]))
    btn_anti.on_clicked(lambda _: set_sliders([0.05, 0.7, 0.7, 0.05]))
    btn_uniform.on_clicked(lambda _: set_sliders([0.5, 0.5, 0.5, 0.5]))
    btn_product.on_clicked(lambda _: set_sliders([0.8, 0.6, 0.4, 0.3]))

    def update(val):
        amps = [s.val for s in sliders]
        if sum(a ** 2 for a in amps) < 1e-10:
            return  # avoid zero state

        result = semantic_circuit(amps, seed=42)
        state = result["state"]
        ev = result["expectation_values"]
        s_val = result["s_value"]

        # Normalized amplitudes
        normed = np.abs(state)
        for bar_rect, a in zip(amp_bars, normed):
            bar_rect.set_height(a)

        # EVs
        ev_vals = [ev["A_B"], ev["A_B_prime"], ev["A_prime_B"], ev["A_prime_B_prime"]]
        for bar_rect, v in zip(bars_ev, ev_vals):
            bar_rect.set_height(v)

        # Pie
        ax_pie.clear()
        ax_pie.set_title("Measurement Outcomes (1k shots)")
        counts = result["measurement_counts"]
        labels_pie = list(counts.keys())
        sizes = list(counts.values())
        if sum(sizes) > 0:
            ax_pie.pie(sizes, labels=labels_pie, colors=amp_colors[:len(labels_pie)],
                      autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})

        # Info
        violation = "YES" if result["violation"] else "no"
        color = "green" if result["violation"] else "red"
        info_text.set_text(
            f"S-value:        {s_val:+.4f}\n"
            f"CHSH violation: {violation}\n"
            f"Tsirelson frac: {result['tsirelson_fraction']:.4f}\n"
            f"Concurrence:    {result['concurrence']:.4f}\n"
            f"Ent. entropy:   {result['entanglement_entropy']:.4f} bits\n\n"
            f"Interpretation:\n"
            + ("  Meanings are entangled!" if result["violation"]
               else "  Meanings behave classically")
        )
        info_text.set_color(color)

        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)
    update(None)
    plt.show(block=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Plot 5: 2D S-value heatmap (chroptiks plot2dhist)
# ═══════════════════════════════════════════════════════════════════════

def plot_s_heatmap():
    """Full 2D heatmap of S-value over (b0, b1) angle space using chroptiks."""

    print("  Computing S-value heatmap (may take a few seconds)...")
    state = bell_state("phi_plus")
    A0, A1 = alice_operators(0, np.pi / 2)

    n = 120
    b0_vals = np.linspace(0, 2 * np.pi, n)
    b1_vals = np.linspace(0, 2 * np.pi, n)
    B0_grid, B1_grid = np.meshgrid(b0_vals, b1_vals)

    S_grid = np.zeros_like(B0_grid)
    for i in range(n):
        for j in range(n):
            B0, B1 = bob_operators(B0_grid[i, j], B1_grid[i, j])
            S_grid[i, j] = chsh_s_value(state, A0, A1, B0, B1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(np.degrees(B0_grid), np.degrees(B1_grid), S_grid,
                        cmap="RdBu_r", vmin=-3, vmax=3, shading="auto")
    plt.colorbar(im, ax=ax, label="S-value")

    # Mark violation regions
    ax.contour(np.degrees(B0_grid), np.degrees(B1_grid), np.abs(S_grid),
               levels=[2.0], colors=["red"], linewidths=[2], linestyles=["--"])
    ax.contour(np.degrees(B0_grid), np.degrees(B1_grid), np.abs(S_grid),
               levels=[2 * np.sqrt(2) - 0.01], colors=["lime"], linewidths=[2], linestyles=[":"])

    # Mark optimal point
    ax.plot(45, 135, "w*", markersize=20, markeredgecolor="k", markeredgewidth=1.5,
            label=f"Optimal (b₀=45°, b₁=135°)")

    ax.set_xlabel("Bob b₀ angle (degrees)", fontsize=12)
    ax.set_ylabel("Bob b₁ angle (degrees)", fontsize=12)
    ax.set_title("CHSH S-value Landscape for |Φ+⟩\n"
                 "Alice: a₀=0°, a₁=90° | Red dashes: |S|=2 | Green: Tsirelson",
                 fontsize=13)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show(block=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Plot 6: Entanglement measures landscape (chroptiks scatter)
# ═══════════════════════════════════════════════════════════════════════

def plot_entanglement_landscape():
    """Scatter: concurrence vs S-value across random states, colored by entropy."""

    print("  Sampling 2000 random 2-qubit states...")
    rng = np.random.default_rng(42)
    n = 2000
    concs, s_vals, entropies = [], [], []

    for _ in range(n):
        # Random state: parameterized + random phase
        theta = rng.uniform(0, np.pi / 2)
        phi = rng.uniform(0, 2 * np.pi)
        psi = parameterized_entangled_state(theta, phi)
        concs.append(concurrence(psi))
        s_vals.append(abs(chsh_s_value(psi)))
        entropies.append(entanglement_entropy(psi))

    concs = np.array(concs)
    s_vals = np.array(s_vals)
    entropies = np.array(entropies)

    fig, ax = makefig()
    sc = ax.scatter(concs, s_vals, c=entropies, cmap="plasma", s=8, alpha=0.6, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Entanglement Entropy (bits)")

    ax.axhline(y=2, color="red", linestyle="--", linewidth=2, label="Classical bound |S|=2")
    ax.axhline(y=2 * np.sqrt(2), color="green", linestyle=":", linewidth=2, label="Tsirelson bound")

    ax.set_xlabel("Concurrence", fontsize=12)
    ax.set_ylabel("|S| value", fontsize=12)
    ax.set_title("Entanglement Measures Landscape\n"
                 "2000 parameterized states, colored by entropy", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    plt.ion()

    print("\nqstk.qc Interactive Explorer")
    print("=" * 50)
    print("\nOpening 6 interactive plots...\n")

    print("[1/6] CHSH Angle Explorer (4 sliders)")
    f1 = plot_chsh_angle_explorer()

    print("[2/6] Werner Noise Explorer (1 slider)")
    f2 = plot_werner_explorer()

    print("[3/6] Entanglement Parameter Explorer (2 sliders)")
    f3 = plot_entanglement_explorer()

    print("[4/6] Semantic State Explorer (4 sliders + presets)")
    f4 = plot_semantic_explorer()

    print("[5/6] S-value Heatmap")
    f5 = plot_s_heatmap()

    print("[6/6] Entanglement Landscape")
    f6 = plot_entanglement_landscape()

    print("\n" + "=" * 50)
    print("All plots open! Drag sliders to explore.")
    print("Close all windows or Ctrl+C to exit.")
    print("=" * 50)

    plt.show(block=True)
