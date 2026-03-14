"""Core quantum semantic methods for QSTK.

This module re-exports the primary CHSH functions for backwards compatibility.
For full functionality, import from the specific submodules directly.
"""

from .chsh import (
    calculate_s_value,
    check_violation,
    compute_chsh_products,
    compute_chsh_products_binary,
    calculate_expectation_values_direct,
    calculate_expectation_values_density_matrix,
)
