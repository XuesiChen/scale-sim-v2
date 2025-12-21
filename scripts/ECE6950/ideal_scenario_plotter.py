import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Load CSVs
# -----------------------
benchmarks = {
    "ResNet50": "ECE6950_Resnet50_BWlimited/GoogleTPU_v1_ws/COMPUTE_REPORT.csv",
    "DeepBench": "ECE6950_deepbench_BWlimited/GoogleTPU_v1_ws/COMPUTE_REPORT.csv",
    "GPT2": "ECE6950_GPT_BWlimited_improved2/GoogleTPU_v1_ws/COMPUTE_REPORT.csv",
}

results = {}

for name, path in benchmarks.items():
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    total_cycles = df["Total Cycles"].sum()
    stall_cycles = df["Stall Cycles"].sum()
    compute_cycles = total_cycles - stall_cycles

    results[name] = {
        "total": total_cycles,
        "compute": compute_cycles,
        "stall": stall_cycles
    }

# -----------------------
# Prepare data
# -----------------------
labels = list(results.keys())
total = np.array([results[b]["total"] for b in labels])
compute = np.array([results[b]["compute"] for b in labels])
stall = np.array([results[b]["stall"] for b in labels])

compute_pct = compute / total * 100
stall_pct = stall / total * 100

x = np.arange(len(labels))

# -----------------------
# Plot
# -----------------------
# make fontsize 8
plt.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.6), gridspec_kw={"wspace": 0.27, "left": 0.11, "right": 0.98, "top": 0.9, "bottom": 0.17})

# === Left: 100% stacked bar (compute vs stall) ===
axs[0].bar(x, compute_pct, label="Compute", color="darkgreen")
axs[0].bar(x, stall_pct, bottom=compute_pct, label="Stall", color="indianred")

axs[0].set_ylabel("Cycle Breakdown (%)", labelpad=0)
axs[0].set_xticks(x)
axs[0].set_xticklabels(["ResNet50", "Deep\nBench", "GPT2"])
axs[0].set_ylim(0, 100)
axs[0].tick_params(pad=0)

axs[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2))

# === Right: total vs compute-only cycles ===
axs[1].plot(labels, total, marker="o", label="Reality", linewidth=2)
axs[1].plot(labels, compute, marker="o", linestyle="--",
            label="Ideal", linewidth=2)

axs[1].set_ylabel("Cycles", labelpad=0)
axs[1].legend(frameon=False, ncol=1, loc="upper center", bbox_to_anchor=(0.7, 0.6))
axs[1].set_xticklabels(["ResNet50", "Deep\nBench", "GPT2"])
axs[1].tick_params(pad=0)

# -----------------------
# Finalize
# -----------------------
plt.tight_layout()
plt.savefig("scripts/ECE6950/figs/cycle_breakdown_and_ideal.png", dpi=300)
# plt.show()



# Plot
# plt.figure(figsize=(16, 5))

# # Non-stall portion
# plt.bar(
#     layer_id,
#     compute_cycles,
#     label="Compute (non-stall) cycles",
#     color='darkgreen'
# )

# # Stall portion stacked on top
# plt.bar(
#     layer_id,
#     stall_cycles,
#     bottom=compute_cycles,
#     label="Stall cycles",
#     color='indianred'
# )

# plt.xlabel("Layer ID")
# plt.ylabel("Cycles")
# plt.title("DeepBench on TPU v1")

# plt.legend()
# plt.tight_layout()

# plt.savefig("scripts/ECE6950/figs/ideal_scenario_plot.png", dpi=300)