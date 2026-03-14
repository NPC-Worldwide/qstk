"""Quantum Semantic Toolkit (QSTK)

Core package for quantum semantic methods based on npcpy.
Provides reusable building blocks for Bell test experiments,
decoherence analysis, semantic trajectory analysis, orbital dynamics,
and Feynman-Kac agent simulations.
"""

from .chsh import (
    calculate_s_value,
    check_violation,
    compute_chsh_products,
    compute_chsh_products_binary,
    calculate_expectation_values_direct,
    calculate_expectation_values_density_matrix,
)
from .personas import generate_persona, get_persona_prompt, create_personas_pool
from .passages import prepare_passages
from .statistics import calculate_agreement_significance_combinatorial
from .results import (
    init_csv,
    append_csv_row,
    append_csv_row_raw,
    find_latest_csv,
    load_previous_trials,
    aggregate_results,
    generate_timestamp,
)
from .grid import get_param_grid, build_sweep_configs, sweep_summary
from .arrays import create_bell_array, create_bell_meshgrid
