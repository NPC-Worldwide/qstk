"""Feynman-Kac inspired agent-based simulation framework.

Models agent populations evolving via stochastic differential equations
with drift (personality), diffusion (randomness), and potential (survival)
components.
"""

from typing import Dict, Any, List, Optional, Callable

import numpy as np


class Agent:
    """An agent with personality-driven stochastic state evolution.

    Parameters
    ----------
    agent_id : any
        Unique identifier for this agent.
    initial_state : dict
        Initial state variables (e.g. {"opinion": 0.5, "energy": 0.8}).
    personality_params : dict
        Personality traits that govern drift and diffusion.
    environment : Environment
        Shared environment reference.
    drift_fn : callable, optional
        Custom drift function(state, personality, environment, t) -> dict.
    diffusion_fn : callable, optional
        Custom diffusion function(state, personality, environment, t) -> dict.
    potential_fn : callable, optional
        Custom potential function(state, personality, environment, t) -> float.
    """

    def __init__(
        self,
        agent_id,
        initial_state: Dict[str, float],
        personality_params: Dict[str, float],
        environment: "Environment",
        drift_fn: Optional[Callable] = None,
        diffusion_fn: Optional[Callable] = None,
        potential_fn: Optional[Callable] = None,
        state_bounds: Optional[Dict[str, tuple]] = None,
    ):
        self.id = agent_id
        self.state = initial_state.copy()
        self.personality = personality_params
        self.environment = environment
        self.history: List[Dict[str, float]] = [initial_state.copy()]
        self._drift_fn = drift_fn
        self._diffusion_fn = diffusion_fn
        self._potential_fn = potential_fn
        self.state_bounds = state_bounds or {}

    def drift(self, state: Dict[str, float], t: float) -> Dict[str, float]:
        """Deterministic drift component."""
        if self._drift_fn:
            return self._drift_fn(state, self.personality, self.environment, t)

        drift = {}
        if "opinion" in state:
            opinion_drift = self.personality.get("bias", 0) - state["opinion"]
            social = self.environment.get_social_influence(self, t)
            drift["opinion"] = (
                self.personality.get("opinion_adaptability", 0.05) * opinion_drift
                + self.personality.get("social_susceptibility", 0.1) * social
            )
        if "energy" in state:
            drift["energy"] = (
                self.personality.get("energy_recovery_rate", 0.05)
                - self.environment.get_energy_consumption(self, t)
            )
        return drift

    def diffusion(self, state: Dict[str, float], t: float) -> Dict[str, float]:
        """Stochastic diffusion component."""
        if self._diffusion_fn:
            return self._diffusion_fn(state, self.personality, self.environment, t)

        diff = {}
        if "opinion" in state:
            diff["opinion"] = self.personality.get("opinion_volatility", 0.1)
        if "energy" in state:
            diff["energy"] = self.personality.get("energy_volatility", 0.02)
        return diff

    def potential(self, state: Dict[str, float], t: float) -> float:
        """Potential function V(x,t) governing survival probability."""
        if self._potential_fn:
            return self._potential_fn(state, self.personality, self.environment, t)

        risk = 0.0
        if "energy" in state:
            risk += max(0, 0.05 * (1 - state["energy"]))
        if "opinion" in state:
            consensus = self.environment.consensus_value("opinion", t)
            alignment = 1 - abs(state["opinion"] - consensus)
            happiness = 0.7 * alignment + 0.3 * state.get("energy", 0.5)
            risk += max(0, 0.1 * (1 - happiness))
        return risk

    def update_state(self, dt: float, t: float) -> bool:
        """Update state via Feynman-Kac SDE. Returns False if agent is removed."""
        d = self.drift(self.state, t)
        sigma = self.diffusion(self.state, t)
        dW = {key: np.random.normal(0, np.sqrt(dt)) for key in sigma}

        new_state = {}
        for key in self.state:
            det = d.get(key, 0) * dt
            stoch = sigma.get(key, 0) * dW.get(key, 0)
            new_state[key] = self.state[key] + det + stoch

        # Apply bounds
        for key, (lo, hi) in self.state_bounds.items():
            if key in new_state:
                new_state[key] = max(lo, min(hi, new_state[key]))

        # Default bounds for known state variables
        if "opinion" in new_state and "opinion" not in self.state_bounds:
            new_state["opinion"] = max(-1, min(1, new_state["opinion"]))
        if "energy" in new_state and "energy" not in self.state_bounds:
            new_state["energy"] = max(0, min(1, new_state["energy"]))

        # Survival check
        V = self.potential(new_state, t)
        survival_prob = np.exp(-V * dt)
        if np.random.random() > survival_prob:
            return False

        self.state = new_state
        self.history.append(new_state.copy())
        return True


class Environment:
    """Shared environment for agent population simulation.

    Parameters
    ----------
    external_factors : dict, optional
        Time-dependent external forces.
        Format: {"factor_name": {time: value, ...}}
    new_agent_fn : callable, optional
        Function(environment, t) -> Agent or None for spawning new agents.
    """

    def __init__(
        self,
        external_factors: Optional[Dict[str, Dict[float, float]]] = None,
        new_agent_fn: Optional[Callable] = None,
    ):
        self.agents: List[Agent] = []
        self.external_factors = external_factors or {}
        self.time = 0.0
        self.history: List[Dict[str, Any]] = []
        self._new_agent_fn = new_agent_fn

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def get_social_influence(self, agent: Agent, t: float) -> float:
        """Distance-weighted social influence from other agents."""
        if not self.agents or "opinion" not in agent.state:
            return 0.0
        opinions = [a.state["opinion"] for a in self.agents]
        agent_op = agent.state["opinion"]
        weights = [np.exp(-3 * abs(op - agent_op)) for op in opinions]
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return sum(w * o for w, o in zip(weights, opinions)) / total_w - agent_op

    def get_energy_consumption(self, agent: Agent, t: float) -> float:
        """Energy consumption rate based on agent and environment."""
        base = 0.05
        vol = 0.1 * agent.personality.get("opinion_volatility", 0.1)
        ext = self.get_external_factor("difficulty", t) * 0.1
        return base + vol + ext

    def get_external_factor(self, name: str, t: float) -> float:
        """Linearly interpolate an external factor at time t."""
        if name not in self.external_factors:
            return 0.0
        data = self.external_factors[name]
        times = sorted(data.keys())
        if t <= times[0]:
            return data[times[0]]
        if t >= times[-1]:
            return data[times[-1]]
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                t1, t2 = times[i], times[i + 1]
                v1, v2 = data[t1], data[t2]
                return v1 + (v2 - v1) * (t - t1) / (t2 - t1)
        return 0.0

    def consensus_value(self, key: str, t: float) -> float:
        """Mean value of a state variable across all agents."""
        vals = [a.state[key] for a in self.agents if key in a.state]
        return sum(vals) / len(vals) if vals else 0.0

    def simulate(
        self,
        total_time: float,
        dt: float,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run the simulation.

        Parameters
        ----------
        total_time : float
            Total simulation time.
        dt : float
            Time step.
        verbose : bool
            Print progress.

        Returns
        -------
        List of per-step summary dicts (also stored in self.history).
        """
        n_steps = int(total_time / dt)

        for step in range(n_steps):
            t = step * dt
            self.time = t

            # Record state
            state_keys = set()
            for a in self.agents:
                state_keys.update(a.state.keys())

            snapshot = {"time": t, "num_agents": len(self.agents)}
            for key in state_keys:
                vals = [a.state[key] for a in self.agents if key in a.state]
                if vals:
                    snapshot[f"mean_{key}"] = float(np.mean(vals))
                    snapshot[f"var_{key}"] = float(np.var(vals))
            self.history.append(snapshot)

            # Update agents
            to_remove = []
            for i, agent in enumerate(self.agents):
                if not agent.update_state(dt, t):
                    to_remove.append(i)
            for i in sorted(to_remove, reverse=True):
                del self.agents[i]

            # Spawn new agents
            if self._new_agent_fn:
                new_agent = self._new_agent_fn(self, t)
                if new_agent is not None:
                    self.add_agent(new_agent)

            if verbose and step % max(1, n_steps // 20) == 0:
                print(f"  t={t:.1f} agents={len(self.agents)}")

        return self.history
