"""Orbital dynamics analysis for semantic trajectories in embedding space.

Computes orbital elements, attractor orbits, Lyapunov exponents,
and Goldilocks bands from embedding trajectories.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


class OrbitType(Enum):
    CIRCULAR = "circular"
    ELLIPTICAL = "elliptical"
    PARABOLIC = "parabolic"
    HYPERBOLIC = "hyperbolic"
    DECAYING = "decaying"
    EXPANDING = "expanding"
    CHAOTIC = "chaotic"


@dataclass
class OrbitalElements:
    eccentricity: float
    semi_major_axis: float
    periapsis: float
    apoapsis: float
    specific_energy: float
    angular_momentum: float
    orbital_period: Optional[float]
    orbit_type: OrbitType
    is_bound: bool
    stability_index: float


@dataclass
class AttractorOrbit:
    attractor_id: int
    attractor_centroid: np.ndarray
    orbital_elements: OrbitalElements
    visit_times: List[int]
    residence_times: List[int]
    escape_velocity: float
    current_velocity: float


@dataclass
class TrajectoryDynamics:
    positions_2d: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    angular_momentum: np.ndarray
    lyapunov_exponent: float
    is_chaotic: bool


@dataclass
class GoldilocksBand:
    temp_lower: float
    temp_upper: float
    avg_eccentricity: float
    avg_bound_fraction: float
    avg_attractor_visits: float
    creativity_score: float


def project_to_orbital_plane(
    embeddings: np.ndarray,
    attractor_centroid: np.ndarray,
) -> Tuple[np.ndarray, Any]:
    """Project high-dimensional embeddings to 2D orbital plane via PCA.

    Returns (projected_2d, pca_object).
    """
    from sklearn.decomposition import PCA
    centered = embeddings - attractor_centroid
    pca = PCA(n_components=2)
    projected = pca.fit_transform(centered)
    return projected, pca


def compute_velocities(positions: np.ndarray) -> np.ndarray:
    """Compute velocity vectors from a position time series."""
    if len(positions) < 2:
        return np.zeros_like(positions)
    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]
    velocities[0] = velocities[1]
    return velocities


def compute_accelerations(velocities: np.ndarray) -> np.ndarray:
    """Compute acceleration vectors from a velocity time series."""
    if len(velocities) < 2:
        return np.zeros_like(velocities)
    accelerations = np.zeros_like(velocities)
    accelerations[1:] = velocities[1:] - velocities[:-1]
    accelerations[0] = accelerations[1]
    return accelerations


def compute_trajectory_dynamics(positions_2d: np.ndarray) -> TrajectoryDynamics:
    """Compute full dynamics from a 2D position trajectory.

    Includes velocities, accelerations, energies, angular momentum,
    and Lyapunov exponent estimation.
    """
    velocities = compute_velocities(positions_2d)
    accelerations = compute_accelerations(velocities)

    speeds = np.linalg.norm(velocities, axis=1)
    kinetic_energy = 0.5 * speeds ** 2

    distances = np.linalg.norm(positions_2d, axis=1)
    distances = np.where(distances < 1e-10, 1e-10, distances)
    potential_energy = -1.0 / distances

    total_energy = kinetic_energy + potential_energy

    # 2D angular momentum (scalar: x*vy - y*vx)
    angular_momentum = (
        positions_2d[:, 0] * velocities[:, 1]
        - positions_2d[:, 1] * velocities[:, 0]
    )

    lyapunov = estimate_lyapunov(positions_2d)
    is_chaotic = lyapunov > 0.01

    return TrajectoryDynamics(
        positions_2d=positions_2d,
        velocities=velocities,
        accelerations=accelerations,
        kinetic_energy=kinetic_energy,
        potential_energy=potential_energy,
        total_energy=total_energy,
        angular_momentum=angular_momentum,
        lyapunov_exponent=lyapunov,
        is_chaotic=is_chaotic,
    )


def estimate_lyapunov(positions: np.ndarray, n_neighbors: int = 5) -> float:
    """Estimate the largest Lyapunov exponent from a position trajectory.

    Uses a simplified nearest-neighbor divergence method.
    """
    if len(positions) < 20:
        return 0.0

    divergences = []
    for i in range(len(positions) - n_neighbors):
        dists = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if len(dists) == 0:
            continue
        nearest_idx = np.argmin(dists) + i + 1
        if nearest_idx + n_neighbors < len(positions) and i + n_neighbors < len(positions):
            initial_dist = max(dists[nearest_idx - i - 1], 1e-10)
            later_dist = np.linalg.norm(
                positions[i + n_neighbors] - positions[nearest_idx + n_neighbors - 1]
            )
            later_dist = max(later_dist, 1e-10)
            divergences.append(np.log(later_dist / initial_dist) / n_neighbors)

    return float(np.mean(divergences)) if divergences else 0.0


def fit_ellipse(points: np.ndarray) -> Dict[str, Any]:
    """Fit an ellipse to 2D points using algebraic fitting.

    Returns dict with center, semi_major, semi_minor, eccentricity,
    angle, and fit_error.
    """
    import scipy.linalg as la

    if len(points) < 5:
        return {
            "center": np.mean(points, axis=0),
            "semi_major": float(np.std(points[:, 0])),
            "semi_minor": float(np.std(points[:, 1])),
            "eccentricity": 0.5,
            "angle": 0.0,
            "fit_error": float("inf"),
        }

    x, y = points[:, 0], points[:, 1]
    D = np.column_stack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)])
    S = D.T @ D

    C = np.zeros((6, 6))
    C[0, 2] = 2
    C[1, 1] = -1
    C[2, 0] = 2

    try:
        eigenvalues, eigenvectors = la.eig(S, C)
        real_mask = np.isreal(eigenvalues)
        pos_mask = np.real(eigenvalues) > 0
        valid_mask = real_mask & pos_mask
        if not np.any(valid_mask):
            valid_mask = real_mask
        if not np.any(valid_mask):
            raise ValueError("No valid eigenvalues")
        valid_indices = np.where(valid_mask)[0]
        min_idx = valid_indices[np.argmin(np.abs(np.real(eigenvalues[valid_indices])))]
        coeffs = np.real(eigenvectors[:, min_idx])
    except Exception:
        coeffs = np.array([1, 0, 1, 0, 0, -np.var(x) - np.var(y)])

    A, B, C_coef, D_coef, E, F = coeffs
    if abs(A) < 1e-10:
        A = 1e-10

    denom = B ** 2 - 4 * A * C_coef
    if abs(denom) < 1e-10:
        denom = -1e-10

    x_center = (2 * C_coef * D_coef - B * E) / denom
    y_center = (2 * A * E - B * D_coef) / denom

    term1 = A + C_coef
    term2 = np.sqrt((A - C_coef) ** 2 + B ** 2)
    num = 2 * (A * E ** 2 + C_coef * D_coef ** 2 - B * D_coef * E + denom * F)

    denom1 = denom * (term1 + term2)
    denom2 = denom * (term1 - term2)

    if denom1 != 0 and num / denom1 > 0:
        semi_major = np.sqrt(abs(num / denom1))
    else:
        semi_major = float(np.std(x))

    if denom2 != 0 and num / denom2 > 0:
        semi_minor = np.sqrt(abs(num / denom2))
    else:
        semi_minor = float(np.std(y))

    if semi_major < semi_minor:
        semi_major, semi_minor = semi_minor, semi_major

    eccentricity = np.sqrt(1 - (semi_minor / max(semi_major, 1e-10)) ** 2) if semi_major > 0 else 0.0
    angle = 0.5 * np.arctan2(B, A - C_coef)

    residuals = A * x ** 2 + B * x * y + C_coef * y ** 2 + D_coef * x + E * y + F
    fit_error = float(np.mean(residuals ** 2))

    return {
        "center": np.array([x_center, y_center]),
        "semi_major": float(semi_major),
        "semi_minor": float(semi_minor),
        "eccentricity": float(np.clip(eccentricity, 0, 1)),
        "angle": float(angle),
        "fit_error": fit_error,
    }


def classify_orbit(eccentricity: float, energy: float) -> OrbitType:
    """Classify an orbit based on eccentricity and specific energy."""
    if eccentricity < 0.05:
        return OrbitType.CIRCULAR
    elif eccentricity < 1.0:
        return OrbitType.ELLIPTICAL
    elif abs(eccentricity - 1.0) < 0.05:
        return OrbitType.PARABOLIC
    elif eccentricity > 1.0:
        return OrbitType.HYPERBOLIC
    elif energy > 0:
        return OrbitType.EXPANDING
    else:
        return OrbitType.DECAYING


def compute_orbital_elements(
    positions_2d: np.ndarray,
    velocities: np.ndarray,
) -> OrbitalElements:
    """Compute Keplerian orbital elements from a 2D trajectory.

    Parameters
    ----------
    positions_2d : array of shape (N, 2)
    velocities : array of shape (N, 2)

    Returns
    -------
    OrbitalElements dataclass.
    """
    ellipse = fit_ellipse(positions_2d)
    ecc = ellipse["eccentricity"]
    a = ellipse["semi_major"]

    speeds = np.linalg.norm(velocities, axis=1)
    distances = np.linalg.norm(positions_2d, axis=1)
    distances = np.where(distances < 1e-10, 1e-10, distances)

    # Specific orbital energy (vis-viva)
    specific_energy = float(np.mean(0.5 * speeds ** 2 - 1.0 / distances))

    # Angular momentum
    L = positions_2d[:, 0] * velocities[:, 1] - positions_2d[:, 1] * velocities[:, 0]
    angular_momentum = float(np.mean(np.abs(L)))

    periapsis = a * (1 - ecc)
    apoapsis = a * (1 + ecc) if ecc < 1.0 else float("inf")

    # Orbital period (Kepler's third law for bound orbits)
    orbital_period = 2 * np.pi * np.sqrt(a ** 3) if ecc < 1.0 and a > 0 else None

    orbit_type = classify_orbit(ecc, specific_energy)
    is_bound = specific_energy < 0

    # Stability index based on energy conservation
    energy_series = 0.5 * speeds ** 2 - 1.0 / distances
    stability_index = 1.0 / (1.0 + float(np.std(energy_series)))

    return OrbitalElements(
        eccentricity=ecc,
        semi_major_axis=a,
        periapsis=periapsis,
        apoapsis=apoapsis,
        specific_energy=specific_energy,
        angular_momentum=angular_momentum,
        orbital_period=orbital_period,
        orbit_type=orbit_type,
        is_bound=is_bound,
        stability_index=stability_index,
    )
