"""
Single Soliton Error and Conservation Analysis
===============================================

Creates 6 plots per method (12 total):

1. Domain investigation - errors
2. Domain investigation - conservation (relative errors)
3. Domain investigation - quantities (actual M, V, E)
4. Node investigation - errors
5. Node investigation - conservation (relative errors)
6. Node investigation - quantities (actual M, V, E)

Each plot uses FacetGrid with c in columns and parameter (L or N) in rows.
"""
# TODO: Compared aliased to non-aliased methods
# %%
# Single soliton diagnostics
# Analyze errors and conservation properties for single soliton solutions.

import pandas as pd
import seaborn as sns

from spectral.utils.plotting import get_repo_root
from spectral.utils.formatting import format_dt_latex

repo_root = get_repo_root()
data_dir = repo_root / "data/A2/ex_d"
save_dir = repo_root / "figures/A2/ex_d"
save_dir.mkdir(parents=True, exist_ok=True)

# %% Load data
print("Loading exercise d) data...")
df_domain_errors = pd.read_parquet(data_dir / "domain_errors.parquet")
df_domain_quantities = pd.read_parquet(data_dir / "domain_quantities.parquet")
df_nodes_errors = pd.read_parquet(data_dir / "nodes_errors.parquet")
df_nodes_quantities = pd.read_parquet(data_dir / "nodes_quantities.parquet")

print(f"  Domain errors: {df_domain_errors.shape}")
print(f"  Domain quantities: {df_domain_quantities.shape}")
print(f"  Nodes errors: {df_nodes_errors.shape}")
print(f"  Nodes quantities: {df_nodes_quantities.shape}")


# %% Helper functions to create plots
TREATMENT_ORDER = ["Aliased", "De-aliased (3/2-rule)"]

TREATMENT_DASHES = {
    "Aliased": "",
    "De-aliased (3/2-rule)": (6, 2.0),
}

ERROR_ORDER = [r"$L^2$ error", r"$L^\infty$ error"]


def plot_errors(df_errors, method, investigation_type, param_name):
    """Create error plot using FacetGrid.

    Parameters
    ----------
    df_errors : DataFrame
        Error data (already in long format)
    method : str
        Method name (RK4, RK3)
    investigation_type : str
        'domain' or 'nodes'
    param_name : str
        'L' for domain, 'N' for nodes
    """
    # Filter for this method
    df_err = df_errors[df_errors["method"] == method].copy()

    # Map error_type to display names
    df_err["error_type"] = df_err["error_type"].map(
        {"l2": r"$L^2$ error", "linf": r"$L^\infty$ error"}
    )
    df_err["error_type"] = pd.Categorical(
        df_err["error_type"], categories=ERROR_ORDER, ordered=True
    )

    if hasattr(df_err["Treatment"], "cat"):
        df_err["Treatment"] = df_err["Treatment"].cat.reorder_categories(
            TREATMENT_ORDER, ordered=True
        )
    else:
        df_err["Treatment"] = pd.Categorical(
            df_err["Treatment"], categories=TREATMENT_ORDER, ordered=True
        )

    # Create relplot with param_name in rows, c in columns
    g = sns.relplot(
        data=df_err,
        x="t",
        y="error",
        hue="error_type",
        hue_order=ERROR_ORDER,
        style="Treatment",
        style_order=TREATMENT_ORDER,
        markers=False,
        dashes=TREATMENT_DASHES,
        row=param_name,
        col="c",
        kind="line",
        facet_kws={"sharex": False, "sharey": False},
    )

    # Set log scale for y-axis
    for ax in g.axes.flat:
        ax.set_yscale("log")

    g.set_axis_labels(r"Time $t$", r"Error")

    # Create mapping of (param_value, c) -> dt for custom titles
    dt_map = df_err.groupby([param_name, "c"])["dt"].first().to_dict()

    # Set custom titles including dt for each facet
    for i, row_val in enumerate(g.row_names):
        for j, col_val in enumerate(g.col_names):
            ax = g.axes[i, j]
            dt_val = dt_map.get((row_val, col_val))
            if dt_val is not None:
                dt_latex = format_dt_latex(dt_val)
                ax.set_title(
                    rf"${param_name}={row_val}$, $c={col_val}$, $\Delta t={dt_latex}$"
                )

    # Title will be set in the main plotting loop with parameters
    return g.fig


def plot_conservation(df_quantities, method, investigation_type, param_name):
    """Create conservation relative error plot using FacetGrid.

    Parameters
    ----------
    df_quantities : DataFrame
        Quantity data (already in long format)
    method : str
        Method name (RK4, RK3)
    investigation_type : str
        'domain' or 'nodes'
    param_name : str
        'L' for domain, 'N' for nodes
    """
    # Filter for this method
    df_cons = df_quantities[df_quantities["method"] == method].copy()

    if hasattr(df_cons["Treatment"], "cat"):
        df_cons["Treatment"] = df_cons["Treatment"].cat.reorder_categories(
            TREATMENT_ORDER, ordered=True
        )
    else:
        df_cons["Treatment"] = pd.Categorical(
            df_cons["Treatment"], categories=TREATMENT_ORDER, ordered=True
        )

    # Create relplot with param_name in rows, c in columns
    g = sns.relplot(
        data=df_cons,
        x="t",
        y="rel_error",
        hue="quantity",
        style="Treatment",
        style_order=TREATMENT_ORDER,
        markers=False,
        row=param_name,
        col="c",
        kind="line",
        facet_kws={"sharex": False, "sharey": False},
        dashes=TREATMENT_DASHES,
    )

    # Add horizontal line at y=0
    for ax in g.axes.flat:
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    g.set_axis_labels(r"Time $t$", r"Relative error")

    # Create mapping of (param_value, c) -> dt for custom titles
    dt_map = df_cons.groupby([param_name, "c"])["dt"].first().to_dict()

    # Set custom titles including dt for each facet
    for i, row_val in enumerate(g.row_names):
        for j, col_val in enumerate(g.col_names):
            ax = g.axes[i, j]
            dt_val = dt_map.get((row_val, col_val))
            if dt_val is not None:
                dt_latex = format_dt_latex(dt_val)
                ax.set_title(
                    rf"${param_name}={row_val}$, $c={col_val}$, $\Delta t={dt_latex}$"
                )

    # Title will be set in the main plotting loop with parameters
    return g.fig


def plot_quantities(df_quantities, method, investigation_type, param_name):
    """Create actual conservation quantities plot using FacetGrid.

    Parameters
    ----------
    df_quantities : DataFrame
        Quantity data (already in long format)
    method : str
        Method name (RK4, RK3)
    investigation_type : str
        'domain' or 'nodes'
    param_name : str
        'L' for domain, 'N' for nodes
    """
    # Filter for this method
    df_qty = df_quantities[df_quantities["method"] == method].copy()

    if hasattr(df_qty["Treatment"], "cat"):
        df_qty["Treatment"] = df_qty["Treatment"].cat.reorder_categories(
            TREATMENT_ORDER, ordered=True
        )
    else:
        df_qty["Treatment"] = pd.Categorical(
            df_qty["Treatment"], categories=TREATMENT_ORDER, ordered=True
        )

    # Create relplot with param_name in rows, c in columns
    g = sns.relplot(
        data=df_qty,
        x="t",
        y="value",
        hue="quantity",
        style="Treatment",
        style_order=TREATMENT_ORDER,
        markers=False,
        row=param_name,
        col="c",
        kind="line",
        facet_kws={"sharex": False, "sharey": False},
        dashes=TREATMENT_DASHES,
    )

    g.set_axis_labels(r"Time $t$", r"Value")

    # Create mapping of (param_value, c) -> dt for custom titles
    dt_map = df_qty.groupby([param_name, "c"])["dt"].first().to_dict()

    # Set custom titles including dt for each facet
    for i, row_val in enumerate(g.row_names):
        for j, col_val in enumerate(g.col_names):
            ax = g.axes[i, j]
            dt_val = dt_map.get((row_val, col_val))
            if dt_val is not None:
                dt_latex = format_dt_latex(dt_val)
                ax.set_title(
                    rf"${param_name}={row_val}$, $c={col_val}$, $\Delta t={dt_latex}$"
                )

    # Title will be set in the main plotting loop with parameters
    return g.fig


# %% Create plots for each method
print("\nCreating plots...")

methods = sorted(df_domain_errors["method"].unique())

for method in methods:
    print(f"\n{method}:")

    # Domain investigation - errors
    fig = plot_errors(df_domain_errors, method, "domain", "L")
    df_method_domain = df_domain_errors[df_domain_errors["method"] == method]
    L_vals = sorted(df_method_domain["L"].unique())
    N_val = df_method_domain["N"].iloc[0]
    T_val = df_method_domain["t"].max()
    fig.suptitle(
        f"{method} - Domain Investigation"
        + "\n"
        + rf"$N = {N_val}$, $L \in [{L_vals[0]:.1f}, {L_vals[-1]:.1f}]$, $T = {T_val:.0f}$",
        y=1.03,
    )
    output = save_dir / f"{method.lower()}_domain_errors.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

    # Domain investigation - conservation (relative errors)
    fig = plot_conservation(df_domain_quantities, method, "domain", "L")
    fig.suptitle(
        f"{method} - Domain Conservation"
        + "\n"
        + rf"$N = {N_val}$, $L \in [{L_vals[0]:.1f}, {L_vals[-1]:.1f}]$, $T = {T_val:.0f}$",
        y=1.03,
    )
    output = save_dir / f"{method.lower()}_domain_conservation.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

    # Domain investigation - quantities (actual values)
    fig = plot_quantities(df_domain_quantities, method, "domain", "L")
    fig.suptitle(
        f"{method} - Domain Quantities"
        + "\n"
        + rf"$N = {N_val}$, $L \in [{L_vals[0]:.1f}, {L_vals[-1]:.1f}]$, $T = {T_val:.0f}$",
        y=1.03,
    )
    output = save_dir / f"{method.lower()}_domain_quantities.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

    # Node investigation - errors
    fig = plot_errors(df_nodes_errors, method, "nodes", "N")
    df_method_nodes = df_nodes_errors[df_nodes_errors["method"] == method]
    N_vals = sorted(df_method_nodes["N"].unique())
    L_val_nodes = df_method_nodes["L"].iloc[0]
    T_val_nodes = df_method_nodes["t"].max()
    fig.suptitle(
        f"{method} - Nodes Investigation"
        + "\n"
        + rf"$L = {L_val_nodes:.1f}$, $N \in [{N_vals[0]}, {N_vals[-1]}]$, $T = {T_val_nodes:.0f}$",
        y=1.03,
    )
    output = save_dir / f"{method.lower()}_nodes_errors.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

    # Node investigation - conservation (relative errors)
    fig = plot_conservation(df_nodes_quantities, method, "nodes", "N")
    fig.suptitle(
        f"{method} - Nodes Conservation"
        + "\n"
        + rf"$L = {L_val_nodes:.1f}$, $N \in [{N_vals[0]}, {N_vals[-1]}]$, $T = {T_val_nodes:.0f}$",
        y=1.03,
    )
    output = save_dir / f"{method.lower()}_nodes_conservation.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

    # Node investigation - quantities (actual values)
    fig = plot_quantities(df_nodes_quantities, method, "nodes", "N")
    fig.suptitle(
        f"{method} - Nodes Quantities"
        + "\n"
        + rf"$L = {L_val_nodes:.1f}$, $N \in [{N_vals[0]}, {N_vals[-1]}]$, $T = {T_val_nodes:.0f}$",
        y=1.03,
    )
    output = save_dir / f"{method.lower()}_nodes_quantities.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

print("\nAll plots created!")
