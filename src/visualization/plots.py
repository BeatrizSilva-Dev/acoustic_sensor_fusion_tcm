import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(
    os.path.join(SCRIPT_PATH, "..", "..", "results")
)
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CROSS_FILE = os.path.join(RESULTS_DIR, "lodo_cross_sensor_20drills.csv")
FUSION_FILE = os.path.join(RESULTS_DIR, "lodo_feature_fusion_20drills.csv")

df_cross = pd.read_csv(CROSS_FILE)
df_cross["method"] = "Cross-Sensor"

df_fusion = pd.read_csv(FUSION_FILE)
df_fusion["method"] = "Feature-Level Fusion"

# Column standardization
for df in [df_cross, df_fusion]:
    df.columns = [c.strip().lower() for c in df.columns]
    df["domain"] = df["domain"].str.capitalize()
    df["n_mics"] = df["n_mics"].str.upper()
    df["config"] = df["config"].str.strip().str.upper()

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 11,
})


# BOXPLOT CROSS-SENSOR VS FUSION (ULTRASONIC 1-MIC)
subset_cross = df_cross[
    (df_cross["domain"] == "Ultrasonic") & (df_cross["n_mics"] == "1MIC")
]
subset_fusion = df_fusion[
    (df_fusion["domain"] == "Ultrasonic") & (df_fusion["n_mics"] == "1MIC")
]

configs_order = ["II", "IE", "EE", "EI"]
box_data = [
    subset_cross[subset_cross["config"] == c]["f1"].values
    for c in configs_order
]
box_data.append(subset_fusion["f1"].values)
box_labels = ["II", "IE", "EE", "EI", "Late Fusion"]

fig, ax = plt.subplots(figsize=(6.2, 4.0))
bp = ax.boxplot(
    box_data,
    tick_labels=box_labels,
    patch_artist=True,
    showmeans=True,
    meanprops=dict(
        marker="^", markeredgecolor="black", markerfacecolor="black"
    ),
    medianprops=dict(color="black", linewidth=1.5),
)

colors = ["#4C72B0", "#56B4E9", "#55A868", "#8172B2", "#D55E00"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.set_ylabel("F1-score", fontweight="bold")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, linestyle=":", alpha=0.6, axis="y")

fig3_png = os.path.join(PLOTS_DIR, "Figure_3_Ultrasonic_1MIC_Distribution.png")
fig3_pdf = os.path.join(PLOTS_DIR, "Figure_3_Ultrasonic_1MIC_Distribution.pdf")
plt.savefig(fig3_png, dpi=300, bbox_inches="tight")
plt.savefig(fig3_pdf, bbox_inches="tight")
plt.close()


#  BOXPLOT SENSOR DENSITY BENCHMARK (1-MIC vs 3-MIC)
conditions = [
    ("Regular", "1MIC", "Audible\n(1-MIC)"),
    ("Regular", "3MIC", "Audible\n(3-MIC)"),
    ("Ultrasonic", "1MIC", "Ultrasonic\n(1-MIC)"),
    ("Ultrasonic", "3MIC", "Ultrasonic\n(3-MIC)"),
]

box4_data = []
box4_labels = []
for dom, mic, lbl in conditions:
    sub = df_fusion[(df_fusion["domain"] == dom) & (df_fusion["n_mics"] == mic)]
    box4_data.append(sub["f1"].values)
    box4_labels.append(lbl)

fig, ax = plt.subplots(figsize=(6.2, 4.0))
bp4 = ax.boxplot(
    box4_data,
    tick_labels=box4_labels,
    patch_artist=True,
    showmeans=True,
    meanprops=dict(
        marker="^", markeredgecolor="black", markerfacecolor="black"
    ),
    medianprops=dict(color="black", linewidth=1.5),
)

colors4 = ["#4C72B0", "#7293CB", "#2CA02C", "#98DF8A"]
for patch, color in zip(bp4["boxes"], colors4):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_ylabel("F1-score", fontweight="bold")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, linestyle=":", alpha=0.6, axis="y")

fig4_png = os.path.join(PLOTS_DIR, "Figure_4_Sensor_Density_Benchmark.png")
fig4_pdf = os.path.join(PLOTS_DIR, "Figure_4_Sensor_Density_Benchmark.pdf")
plt.savefig(fig4_png, dpi=300, bbox_inches="tight")
plt.savefig(fig4_pdf, bbox_inches="tight")
plt.close()


# AGGREGATED BARS: CROSS-SENSOR VS FEATURE FUSION
metrics_names = ["f1", "mcc", "auc"]
metrics_labels = ["F1-score", "MCC", "AUC"]

cross_means = [df_cross[m].mean() for m in metrics_names]
cross_stds = [df_cross[m].std() for m in metrics_names]

fusion_means = [df_fusion[m].mean() for m in metrics_names]
fusion_stds = [df_fusion[m].std() for m in metrics_names]

x = np.arange(len(metrics_labels))
width = 0.32

fig, ax = plt.subplots(figsize=(6.5, 4.4))

rects1 = ax.bar(
    x - width / 2,
    cross_means,
    width,
    yerr=cross_stds,
    label="Cross-Sensor",
    color="#1F77B4",
    edgecolor="black",
    capsize=4,
    error_kw=dict(lw=1.2, capthick=1.2),
    alpha=0.9,
    zorder=2,
)
rects2 = ax.bar(
    x + width / 2,
    fusion_means,
    width,
    yerr=fusion_stds,
    label="Feature-Level Fusion",
    color="#FF7F0E",
    edgecolor="black",
    capsize=4,
    error_kw=dict(lw=1.2, capthick=1.2),
    alpha=0.9,
    zorder=2,
)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(
        f"{h:.3f}",
        xy=(rect.get_x() + rect.get_width() / 2, h),
        xytext=(-6, 4),  
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        zorder=4,
    )

for rect in rects2:
    h = rect.get_height()
    ax.annotate(
        f"{h:.3f}",
        xy=(rect.get_x() + rect.get_width() / 2, h),
        xytext=(6, 4),  
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        zorder=4,
    )

ax.set_ylabel("Mean performance", fontweight="bold")
ax.set_xlabel("Evaluation metric", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0.0, 1.20)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.14),
    ncol=2,
    frameon=False,
    fontsize=9.5,
)
ax.grid(True, linestyle=":", alpha=0.6, axis="y", zorder=1)

fig5_png = os.path.join(PLOTS_DIR, "Figure_5_Aggregate_Comparison.png")
fig5_pdf = os.path.join(PLOTS_DIR, "Figure_5_Aggregate_Comparison.pdf")
plt.savefig(fig5_png, dpi=300, bbox_inches="tight")
plt.savefig(fig5_pdf, bbox_inches="tight")
plt.close()

print(f"Figure 5 successfully saved to: {fig5_pdf}")

print(f"All figures generated successfully in: {PLOTS_DIR}")
