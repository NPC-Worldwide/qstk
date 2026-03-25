"""CSV logging, result aggregation, and resume infrastructure."""

import os
import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

from .chsh import calculate_expectation_values_density_matrix, calculate_s_value


def init_csv(output_dir: str, filename: str, columns: List[str]) -> str:
    """Create a CSV file with headers if it doesn't already exist.

    Returns the full path to the CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(",".join(columns) + "\n")
    return path


def append_csv_row(path: str, row_dict: Dict[str, Any], columns: List[str]) -> None:
    """Append one row to a CSV file, escaping commas and quotes."""
    with open(path, "a") as f:
        vals = []
        for c in columns:
            v = str(row_dict.get(c, "")).replace('"', "'").replace(",", ";")
            vals.append(f'"{v}"')
        f.write(",".join(vals) + "\n")


def append_csv_row_raw(path: str, row_dict: Dict[str, Any], columns: List[str]) -> None:
    """Append one row to a CSV without quoting (for numeric-heavy rows)."""
    with open(path, "a") as f:
        vals = [str(row_dict.get(c, "")) for c in columns]
        f.write(",".join(vals) + "\n")


def find_latest_csv(output_dir: str, pattern: str = "bell_grid_[0-9]*.csv") -> Optional[str]:
    """Find the most recent CSV matching a glob pattern in output_dir."""
    import glob as globmod
    full_pattern = os.path.join(output_dir, pattern)
    files = sorted(globmod.glob(full_pattern))
    return files[-1] if files else None


def load_previous_trials(
    csv_path: str,
    group_cols: Optional[List[str]] = None,
) -> Dict:
    """Load a previous CSV and return completed trial counts per grid point.

    Parameters
    ----------
    csv_path : str
        Path to the raw results CSV.
    group_cols : list of str, optional
        Columns to group by. Defaults to the standard bell grid columns.

    Returns
    -------
    dict keyed by tuple of group values -> count of trials.
    """
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    if group_cols is None:
        group_cols = ["model", "word_pair", "flipped", "temperature", "top_p", "top_k"]

    counts = {}
    for cols_tuple, group in df.groupby(group_cols):
        vals = list(cols_tuple)
        # Normalize flipped column if present
        for i, col in enumerate(group_cols):
            if col == "flipped" and isinstance(vals[i], str):
                vals[i] = vals[i].strip() == "True"
        counts[tuple(vals)] = len(group)

    return counts


def aggregate_results(
    df: pd.DataFrame,
    group_cols: List[str],
    product_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate raw trial data into per-grid-point S-values.

    Parameters
    ----------
    df : DataFrame
        Raw trial results with product term columns.
    group_cols : list of str
        Columns to group by.
    product_cols : list of str, optional
        Names of the four product term columns. Defaults to the standard names.

    Returns
    -------
    DataFrame with one row per group, including S-value and expectation values.
    """
    if product_cols is None:
        product_cols = ["A_B", "A_B_prime", "A_prime_B", "A_prime_B_prime"]

    agg_rows = []
    for keys, group in df.groupby(group_cols):
        complete = group[group["complete"] == True] if "complete" in group.columns else group
        products_list = []
        for _, row in complete.iterrows():
            products_list.append({pc: row[pc] for pc in product_cols})

        exp_vals = calculate_expectation_values_density_matrix(products_list)
        s_val = calculate_s_value(exp_vals)

        row_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row_dict.update({
            "s_value": s_val,
            "n_complete": len(complete),
            "n_trials": len(group),
            "E_AB": exp_vals["A_B"],
            "E_AB_prime": exp_vals["A_B_prime"],
            "E_ApB": exp_vals["A_prime_B"],
            "E_ApBp": exp_vals["A_prime_B_prime"],
            "violation": abs(s_val) > 2.0,
        })
        agg_rows.append(row_dict)

    return pd.DataFrame(agg_rows)


def generate_timestamp() -> str:
    """Return a timestamp string suitable for filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
