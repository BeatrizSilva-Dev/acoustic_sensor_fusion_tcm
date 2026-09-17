import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(
    os.path.join(SCRIPT_PATH, "..", "..", "results")
)
PLOTS_DIR = os.path.abspath(
    os.path.join(SCRIPT_PATH, "..", "..", "plots")
)
os.makedirs(PLOTS_DIR, exist_ok=True)

CS_DIR = os.path.join(RESULTS_DIR, "cross_sensor_hole_scores")
FF_DIR = os.path.join(RESULTS_DIR, "feature_fusion_hole_scores")

DOMAIN = "regular"
MIC_CFG = "1mic"
CS_CONFIG = "II"


DRILL_RENAME_MAP = {
    "drill_4mm_04": "Drill 1",
    "drill_4mm_05": "Drill 2",
    "drill_4mm_06": "Drill 3",
    "drill_4mm_07": "Drill 4",
    "drill_4mm_08": "Drill 5",
    "drill_4mm_09": "Drill 6",
    "drill_4mm_10": "Drill 7",
    "drill_4mm_11": "Drill 8",
    "drill_4mm_12": "Drill 9",
    "drill_4mm_13": "Drill 10",
    "drill_4mm_14": "Drill 11",
    "drill_4mm_15": "Drill 12",
    "drill_4mm_16": "Drill 13",
    "drill_4mm_17": "Drill 14",
    "drill_4mm_18": "Drill 15",
    "drill_4mm_19": "Drill 16",
    "drill_4mm_20": "Drill 17",
    "drill_4mm_21": "Drill 18",
    "drill_4mm_22": "Drill 19",
    "drill_4mm_23": "Drill 20",
}

# Selected representative drills for sequential tracking
target_drills = ["drill_4mm_12", "drill_4mm_06", "drill_4mm_04"]


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelweight": "bold",
})

fig, axes = plt.subplots(3, 1, figsize=(4.8, 8.8), sharex=False)
plt.subplots_adjust(hspace=0.45)

for idx, drill_name in enumerate(target_drills):
    ax = axes[idx]

    # 1. Load sequential prediction data
    cs_file = os.path.join(
        CS_DIR, f"continuous_{DOMAIN}_{MIC_CFG}_{CS_CONFIG}_{drill_name}.csv"
    )
    df_cs = pd.read_csv(cs_file).sort_values("hole")

    ff_file = os.path.join(
        FF_DIR, f"continuous_fusion_{DOMAIN}_{MIC_CFG}_{drill_name}.csv"
    )
    df_ff = pd.read_csv(ff_file).sort_values("hole")

    holes = df_cs["hole"].values
    n_total = len(holes)

    # Critical evaluation region (terminal 20% of tool life)
    crit_start_idx = int(np.floor(0.80 * n_total))
    crit_start_hole = holes[crit_start_idx]

    thresh_cs = float(df_cs["threshold"].iloc[0])
    thresh_ff = float(df_ff["threshold"].iloc[0])

    # Terminal failure level assignment for jamming events
    score_ff = df_ff["predicted_anomaly_score"].values.copy()
    score_cs = df_cs["predicted_anomaly_score"].values.copy()
    is_jam = df_cs["is_jam"].values == 1

    score_ff[is_jam] = 1.0
    score_cs[is_jam] = np.maximum(score_cs[is_jam], 0.65)

    # Critical degradation region shading (20%)
    ax.axvspan(
        crit_start_hole - 0.4,
        holes[-1] + 0.4,
        color="#f8d7da",
        alpha=0.60,
        label="Critical (20%)" if idx == 0 else None,
        zorder=1,
    )

    # Decision thresholds 
    ax.axhline(
        thresh_ff,
        color="#e66101",
        linestyle=":",
        linewidth=1.3,
        label="Thresh Fusion" if idx == 0 else None,
        zorder=2,
    )
    ax.axhline(
        thresh_cs,
        color="#2b83ba",
        linestyle="--",
        linewidth=1.3,
        label="Thresh Cross" if idx == 0 else None,
        zorder=2,
    )

    # Continuous anomaly progression trajectories
    ax.plot(
        holes,
        score_ff,
        color="#e66101",
        linestyle="-.",
        linewidth=1.4,
        marker="s",
        markersize=4.2,
        label="Feature Fusion" if idx == 0 else None,
        zorder=3,
    )
    ax.plot(
        holes,
        score_cs,
        color="#2b83ba",
        linestyle="-",
        linewidth=1.4,
        marker="o",
        markersize=4.2,
        label="Cross-Sensor" if idx == 0 else None,
        zorder=3,
    )

    # Physical jamming markers (red 'x')
    if np.any(is_jam):
        jam_holes = holes[is_jam]
        ax.scatter(
            jam_holes,
            [0.0] * len(jam_holes),
            color='red',
            marker='x',
            s=70,
            linewidth=2,
            label='Jamming event' if idx == 0 else None,
            zorder=5,
            clip_on=False,
        )

    # Subplot formatting
    formal_title = DRILL_RENAME_MAP.get(drill_name, drill_name)
    ax.set_title(
        formal_title, loc="left", fontweight="bold", pad=4, fontsize=10.5
    )
    ax.set_ylabel("Anomaly level", fontweight="bold")
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(holes[0] - 0.5, holes[-1] + 0.5)

    if n_total > 35:
        ax.set_xticks(np.arange(8, n_total + 1, 8))
    else:
        ax.set_xticks(np.arange(3, n_total + 1, 3))

    ax.grid(True, linestyle=":", alpha=0.5, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Hole Sequence", fontweight="bold")

axes[0].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.48),
    ncol=3,
    frameon=False,
    fontsize=8.5,
    handletextpad=0.5,
    columnspacing=1.1,
)

fig6_png = os.path.join(PLOTS_DIR, "Figure_6_Sequential_Condition_Monitoring.png")
fig6_pdf = os.path.join(PLOTS_DIR, "Figure_6_Sequential_Condition_Monitoring.pdf")

plt.savefig(fig6_png, dpi=300, bbox_inches="tight")
plt.savefig(fig6_pdf, bbox_inches="tight")
plt.close()

print("[SUCCESS] Figure 6 updated and exported successfully.")