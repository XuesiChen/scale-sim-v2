import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# make fontsize 8
plt.rcParams.update({'font.size': 9})

# =========================
# User config
# =========================
workload = "GPT"
COMPUTE_CSV = f"ECE6950_{workload}_BWlimited/GoogleTPU_v1_ws/COMPUTE_REPORT.csv"
MEMORY_CSV  = f"ECE6950_{workload}_BWlimited/GoogleTPU_v1_ws/DETAILED_ACCESS_REPORT.csv"

OUT_DIR = Path("scripts/ECE6950/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot as % of total layer cycles (recommended for papers).
# Set False for raw cycles.
PLOT_AS_PERCENT_OF_TOTAL = True

# =========================
# Helpers
# =========================
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def clip_interval(start, stop, lo, hi):
    """
    Clips [start, stop] into [lo, hi].
    Handles negative starts and stop=-1.
    Returns (s, e) with s<=e; zero-length means no activity in-window.
    """
    s = _safe_float(start)
    e = _safe_float(stop)

    if np.isnan(s) or np.isnan(e):
        return (lo, lo)

    if e == -1:
        return (lo, lo)

    s = max(lo, min(s, hi))
    e = max(lo, min(e, hi))

    if e < s:
        return (lo, lo)
    return (s, e)

def interval_len(iv):
    s, e = iv
    return max(0.0, e - s)

def union_len(intervals):
    """Length of union of a list of [s,e] intervals."""
    ints = [(s, e) for (s, e) in intervals if e > s]
    if not ints:
        return 0.0

    ints.sort(key=lambda x: x[0])
    merged = []
    cs, ce = ints[0]

    for s, e in ints[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e

    merged.append((cs, ce))
    return sum(e - s for s, e in merged)

def stacked_bar(ax, x, stacks, labels, title, ylabel):
    bottom = np.zeros(len(x), dtype=float)
    for y, lab in zip(stacks, labels):
        ax.bar(x, y, bottom=bottom, label=lab)
        bottom += np.array(y)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("LayerID")
    ax.legend(ncols=min(3, len(labels)), frameon=False)
    ax.set_xticks(x)
    ax.tick_params(axis="x", rotation=0)

# =========================
# Load CSVs
# =========================
comp = pd.read_csv(COMPUTE_CSV)
comp.columns = [c.strip() for c in comp.columns]
comp = comp.rename(columns={
    "Total Cycles": "TotalCycles",
    "Stall Cycles": "StallCycles",
})
comp["LayerID"] = comp["LayerID"].astype(int)

mem = pd.read_csv(MEMORY_CSV)
mem.columns = [c.strip() for c in mem.columns]
mem["LayerID"] = mem["LayerID"].astype(int)

df = (
    pd.merge(
        comp[["LayerID", "TotalCycles", "StallCycles"]],
        mem,
        on="LayerID",
        how="inner"
    )
    .sort_values("LayerID")
    .reset_index(drop=True)
)

# =========================
# Column definitions
# =========================
SRAM_COLS = {
    "IFMAP": ("SRAM IFMAP Start Cycle", "SRAM IFMAP Stop Cycle"),
    "Filter": ("SRAM Filter Start Cycle", "SRAM Filter Stop Cycle"),
    "OFMAP": ("SRAM OFMAP Start Cycle", "SRAM OFMAP Stop Cycle"),
}

DRAM_COLS = {
    "IFMAP": ("DRAM IFMAP Start Cycle", "DRAM IFMAP Stop Cycle"),
    "Filter": ("DRAM Filter Start Cycle", "DRAM Filter Stop Cycle"),
    "OFMAP": ("DRAM OFMAP Start Cycle", "DRAM OFMAP Stop Cycle"),
}

# =========================
# Compute breakdowns
# =========================
A_ifmap, A_filter, A_ofmap = [], [], []
B_dram, B_sram = [], []

layer_ids = df["LayerID"].to_numpy()

for _, row in df.iterrows():
    total = float(row["TotalCycles"])
    stall = float(row["StallCycles"])

    lo, hi = 0.0, max(total, 0.0)

    op_span = {}
    for op in ["IFMAP", "Filter", "OFMAP"]:
        s_s, s_e = SRAM_COLS[op]
        d_s, d_e = DRAM_COLS[op]

        sram_iv = clip_interval(row[s_s], row[s_e], lo, hi)
        dram_iv = clip_interval(row[d_s], row[d_e], lo, hi)
        op_span[op] = interval_len(sram_iv) + interval_len(dram_iv)

    span_sum = sum(op_span.values())
    if stall > 0 and span_sum > 0:
        A_ifmap.append(stall * op_span["IFMAP"] / span_sum)
        A_filter.append(stall * op_span["Filter"] / span_sum)
        A_ofmap.append(stall * op_span["OFMAP"] / span_sum)
    else:
        A_ifmap.append(0.0)
        A_filter.append(0.0)
        A_ofmap.append(0.0)

    sram_intervals, dram_intervals = [], []
    for op in ["IFMAP", "Filter", "OFMAP"]:
        s_s, s_e = SRAM_COLS[op]
        d_s, d_e = DRAM_COLS[op]
        sram_intervals.append(clip_interval(row[s_s], row[s_e], lo, hi))
        dram_intervals.append(clip_interval(row[d_s], row[d_e], lo, hi))

    dram_u = union_len(dram_intervals)
    if stall > 0 and hi > 0:
        f_dram = min(1.0, dram_u / hi)
        B_dram.append(stall * f_dram)
        B_sram.append(stall * (1.0 - f_dram))
    else:
        B_dram.append(0.0)
        B_sram.append(0.0)

# =========================
# Normalize if requested
# =========================
if PLOT_AS_PERCENT_OF_TOTAL:
    totals = np.where(df["TotalCycles"].to_numpy() > 0,
                      df["TotalCycles"].to_numpy(), 1.0)

    A_ifmap = np.nan_to_num(np.array(A_ifmap) / totals * 100)
    A_filter = np.nan_to_num(np.array(A_filter) / totals * 100)
    A_ofmap  = np.nan_to_num(np.array(A_ofmap)  / totals * 100)
    B_dram   = np.nan_to_num(np.array(B_dram)   / totals * 100)
    B_sram   = np.nan_to_num(np.array(B_sram)   / totals * 100)

    ylab_A = ylab_B = "Stall (% of total cycles)"
else:
    ylab_A = ylab_B = "Stall cycles"


# # Instead, do percent of STALL (operand breakdown adds to 100%)
# stall = df["StallCycles"].to_numpy().astype(float)
# stall_safe = np.where(stall > 0, stall, 1.0)

# A_ifmap = np.array(A_ifmap) / stall_safe * 100
# A_filter = np.array(A_filter) / stall_safe * 100
# A_ofmap  = np.array(A_ofmap)  / stall_safe * 100
# ylab_A = "Stall breakdown (% of stall cycles)"

# =========================
# Compute vs stall (raw)
# =========================
stall_cycles = df["StallCycles"].to_numpy()
compute_cycles = df["TotalCycles"].to_numpy() - stall_cycles

# =========================
# Combined figure
# =========================
fig, axs = plt.subplots(3, 1, figsize=(7, 5), sharex=True)

# (a) compute vs stall
axs[0].bar(layer_ids, compute_cycles, label="Compute", color="darkgreen")
axs[0].bar(layer_ids, stall_cycles, bottom=compute_cycles, label="Stall", color="indianred")
axs[0].set_ylabel("Cycles", labelpad=0)
axs[0].legend(frameon=False, ncols=2, bbox_to_anchor=(0.5, 1.17), loc ="upper center")


# (b) stall by operand
axs[1].bar(layer_ids, A_ifmap, label="IFMAP")
axs[1].bar(layer_ids, A_filter, bottom=A_ifmap, label="Filter")
axs[1].bar(layer_ids, A_ofmap,
           bottom=np.array(A_ifmap) + np.array(A_filter),
           label="OFMAP")
axs[1].set_ylabel(ylab_A, labelpad=0)
axs[1].set_ylim(0, 100)
axs[1].legend(frameon=False, ncols=3, bbox_to_anchor=(0.5, 1.17), loc="upper center")

# (c) stall by memory
axs[2].bar(layer_ids, B_dram, label="DRAM", color="tan")
axs[2].bar(layer_ids, B_sram, bottom=B_dram, label="SRAM", color="magenta")
axs[2].set_xlabel("Layer ID", labelpad=0)
axs[2].set_ylabel(ylab_B, labelpad=0)
axs[2].set_ylim(0, 100)
axs[2].legend(frameon=False, ncols=2, bbox_to_anchor=(0.5, 1.17), loc="upper center")

plt.subplots_adjust(hspace=0.15, top=0.97, bottom=0.08, left=0.08, right=0.98)
plt.savefig(OUT_DIR / f"{workload}_stall_cycle_breakdown_combined.png", dpi=300)
print(f"Saved: {OUT_DIR / f'{workload}_stall_cycle_breakdown_combined.png'}")

# =========================
out_df = pd.DataFrame({
    "LayerID": layer_ids,
    "TotalCycles": df["TotalCycles"].astype(float).to_list(),
    "StallCycles": df["StallCycles"].astype(float).to_list(),

    # Operand attribution of stall
    "Stall_IFMAP": np.array(A_ifmap, dtype=float),
    "Stall_Filter": np.array(A_filter, dtype=float),
    "Stall_OFMAP": np.array(A_ofmap, dtype=float),

    # Memory-hierarchy attribution of stall
    "Stall_DRAM_attrib": np.array(B_dram, dtype=float),
    "Stall_SRAM_attrib": np.array(B_sram, dtype=float),
})

# Save
csv_path = OUT_DIR / f"{workload}_stall_breakdown_export.csv"
out_df.to_csv(csv_path, index=False)
print(f"Saved breakdown CSV: {csv_path}")
