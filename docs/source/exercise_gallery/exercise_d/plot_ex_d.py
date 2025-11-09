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

# %%
# Single soliton diagnostics
# Analyze errors and conservation properties for single soliton solutions.

import pandas as pd
import seaborn as sns

from spectral.utils.plotting import add_parameter_footer, get_repo_root

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

    # Create relplot with param_name in rows, c in columns
    g = sns.relplot(
        data=df_err,
        x="t",
        y="error",
        hue="error_type",
        row=param_name,
        col="c",
        kind="line",
        height=3.5,
        aspect=1.3,
        facet_kws={"sharex": True, "sharey": False},
    )

    # Set log scale for y-axis
    for ax in g.axes.flat:
        ax.set_yscale("log")

    g.set_axis_labels(r"Time $t$", r"Error")
    g.set_titles(
        row_template=rf"${param_name}$={{row_name}}", col_template=r"$c$={col_name}"
    )

    param_display = "L" if param_name == "L" else "N"
    fixed_param = "N=128" if param_name == "L" else "L=30"
    g.fig.suptitle(
        f"{method} - {investigation_type.capitalize()} Investigation: Errors (varying {param_display}, {fixed_param})",
        y=1.02,
    )

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

    # Create relplot with param_name in rows, c in columns
    g = sns.relplot(
        data=df_cons,
        x="t",
        y="rel_error",
        hue="quantity",
        row=param_name,
        col="c",
        kind="line",
        height=3.5,
        aspect=1.3,
        facet_kws={"sharex": True, "sharey": False},
    )

    # Add horizontal line at y=0
    for ax in g.axes.flat:
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    g.set_axis_labels(r"Time $t$", r"Relative error")
    g.set_titles(
        row_template=rf"${param_name}$={{row_name}}", col_template=r"$c$={col_name}"
    )

    param_display = "L" if param_name == "L" else "N"
    fixed_param = "N=128" if param_name == "L" else "L=30"
    g.fig.suptitle(
        f"{method} - {investigation_type.capitalize()} Investigation: Conservation (varying {param_display}, {fixed_param})",
        y=1.02,
    )

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

    # Create relplot with param_name in rows, c in columns
    g = sns.relplot(
        data=df_qty,
        x="t",
        y="value",
        hue="quantity",
        row=param_name,
        col="c",
        kind="line",
        height=3.5,
        aspect=1.3,
        facet_kws={"sharex": True, "sharey": False},
    )

    g.set_axis_labels(r"Time $t$", r"Value")
    g.set_titles(
        row_template=rf"${param_name}$={{row_name}}", col_template=r"$c$={col_name}"
    )

    param_display = "L" if param_name == "L" else "N"
    fixed_param = "N=128" if param_name == "L" else "L=30"
    g.fig.suptitle(
        f"{method} - {investigation_type.capitalize()} Investigation: Quantities M, V, E (varying {param_display}, {fixed_param})",
        y=1.02,
    )

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
    add_parameter_footer(
        fig, rf"$N = 128$, $L \in [{L_vals[0]:.1f}, {L_vals[-1]:.1f}]$, $T = 20$"
    )
    output = save_dir / f"{method.lower()}_domain_errors.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

    # Domain investigation - conservation (relative errors)
    fig = plot_conservation(df_domain_quantities, method, "domain", "L")
    add_parameter_footer(
        fig, rf"$N = 128$, $L \in [{L_vals[0]:.1f}, {L_vals[-1]:.1f}]$, $T = 20$"
    )
    output = save_dir / f"{method.lower()}_domain_conservation.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

    # Domain investigation - quantities (actual values)
    fig = plot_quantities(df_domain_quantities, method, "domain", "L")
    add_parameter_footer(
        fig, rf"$N = 128$, $L \in [{L_vals[0]:.1f}, {L_vals[-1]:.1f}]$, $T = 20$"
    )
    output = save_dir / f"{method.lower()}_domain_quantities.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

    # Node investigation - errors
    fig = plot_errors(df_nodes_errors, method, "nodes", "N")
    df_method_nodes = df_nodes_errors[df_nodes_errors["method"] == method]
    N_vals = sorted(df_method_nodes["N"].unique())
    add_parameter_footer(
        fig, rf"$L = 30.0$, $N \in [{N_vals[0]}, {N_vals[-1]}]$, $T = 20$"
    )
    output = save_dir / f"{method.lower()}_nodes_errors.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

    # Node investigation - conservation (relative errors)
    fig = plot_conservation(df_nodes_quantities, method, "nodes", "N")
    add_parameter_footer(
        fig, rf"$L = 30.0$, $N \in [{N_vals[0]}, {N_vals[-1]}]$, $T = 20$"
    )
    output = save_dir / f"{method.lower()}_nodes_conservation.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

    # Node investigation - quantities (actual values)
    fig = plot_quantities(df_nodes_quantities, method, "nodes", "N")
    add_parameter_footer(
        fig, rf"$L = 30.0$, $N \in [{N_vals[0]}, {N_vals[-1]}]$, $T = 20$"
    )
    output = save_dir / f"{method.lower()}_nodes_quantities.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"  Saved: {output}")

print("\nAll plots created!")
