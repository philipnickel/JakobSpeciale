"""
Two-Soliton KdV Collision Animation
====================================

Animates the two-soliton collision in the KdV equation showing wave
propagation and interaction.
"""


# %%
# Animation generation
# Create animation of two-soliton collision.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

from spectral.utils.plotting import get_repo_root

repo_root = get_repo_root()
data_dir = repo_root / "data/A2/ex_f"
fig_dir = repo_root / "figures/A2/ex_f"
fig_dir.mkdir(parents=True, exist_ok=True)

parquet_path = data_dir / "kdv_two_soliton.parquet"

output_path = fig_dir / "solution.mp4"

max_frames = 600
max_x_points = 400
fps = 20

print("=" * 60)
print("Exercise f – two-soliton animation")
print("=" * 60)

# %% Load data ---------------------------------------------------------------
if not parquet_path.exists():
    raise FileNotFoundError(
        f"No dataset found at {parquet_path}. Run ex_f_compute.py first."
    )

print(f"Loading parquet data: {parquet_path}")
df = pd.read_parquet(parquet_path)

print(f"Data shape: {df.shape}")

# Metadata for context
metadata_cols = ["dx", "dt", "N", "L", "save_every", "c1", "x01", "c2", "x02"]
metadata = {col: df[col].iloc[0] for col in metadata_cols if col in df.columns}

print("Metadata:")
for key, val in metadata.items():
    print(f"  {key} = {val}")

# %% Shape arrays ------------------------------------------------------------
x_vals = np.sort(df["x"].unique())
t_vals = np.sort(df["t"].unique())

print(f"Unique x count: {len(x_vals)}, unique t count: {len(t_vals)}")

U = (
    df.pivot(index="x", columns="t", values="u")
    .reindex(index=x_vals, columns=t_vals)
    .to_numpy()
)


# %% Downsampling ------------------------------------------------------------
def _select_indices(n: int, max_points: int) -> np.ndarray:
    """Return indices that downsample to at most max_points while keeping endpoints."""
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    stride = int(np.ceil(n / max_points))
    idx = np.arange(0, n, stride, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


idx_x = _select_indices(len(x_vals), max_x_points)
idx_t = _select_indices(len(t_vals), max_frames)

print(
    f"Downsampling: {len(idx_x)} spatial points, {len(idx_t)} time frames (fps={fps})"
)

x_plot = x_vals[idx_x]
U_plot = U[np.ix_(idx_x, idx_t)]
t_plot = t_vals[idx_t]

# Remove frames containing non-finite values (can happen with coarse runs)
finite_mask = np.all(np.isfinite(U_plot), axis=0)
if not np.all(finite_mask):
    removed = np.count_nonzero(~finite_mask)
    print(f"[warn] Removing {removed} frames with non-finite values.")
    U_plot = U_plot[:, finite_mask]
    t_plot = t_plot[finite_mask]

if U_plot.size == 0:
    raise RuntimeError("No valid frames available for animation.")

# %% Animation setup ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
(line,) = ax.plot(x_plot, U_plot[:, 0], lw=1.5)
ax.set_xlim(x_plot[0], x_plot[-1])

u_min = np.min(U_plot)
u_max = np.max(U_plot)
margin = 0.05 * (u_max - u_min if u_max > u_min else 1.0)
ax.set_ylim(u_min - margin, u_max + margin)

title_text = ax.set_title(f"t = {t_plot[0]:.2f}")
ax.set_xlabel(r"Position $x$")
ax.set_ylabel(r"$u(x, t)$")

num_frames = U_plot.shape[1]
print(f"Animation running over {num_frames} frames...")


def update(frame_index: int):
    line.set_ydata(U_plot[:, frame_index])
    title_text.set_text(f"t = {t_plot[frame_index]:.2f}")
    return line, title_text


if num_frames < 2:
    print(
        f"[warn] Only {num_frames} valid frame(s) available. "
        "Animation may be uninformative."
    )

ani = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=1000 / fps,
    blit=True,
)

# %% Save animation ----------------------------------------------------------
print(f"Saving animation → {output_path}")

try:
    # Use FFMpegWriter with optimized settings for speed
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata=dict(artist="matplotlib"),
        bitrate=1800,
        codec="libx264",
        extra_args=["-preset", "ultrafast", "-pix_fmt", "yuv420p"],
    )
    ani.save(output_path, writer=writer)
except Exception as exc:
    raise RuntimeError(f"Failed to save animation: {exc}") from exc

print(f"Saved animation → {output_path}")
print("=" * 60)
print("Animation complete.")
print("=" * 60)
