import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(
    os.path.join(SCRIPT_PATH, "..", "..", "results")
)

CROSS_FILE = os.path.join(RESULTS_DIR, "lodo_cross_sensor_20drills.csv")
FUSION_FILE = os.path.join(RESULTS_DIR, "lodo_feature_fusion_20drills.csv")

CORE_METRICS = ["f1", "mcc", "auc"]
ALL_METRICS = ["f1", "mcc", "auc", "precision", "recall"]


df_cross = pd.read_csv(CROSS_FILE)
df_cross["method"] = "Cross-Sensor"

df_fusion = pd.read_csv(FUSION_FILE)
df_fusion["method"] = "Feature-Level Fusion"

results = pd.concat([df_cross, df_fusion], ignore_index=True)
results.columns = [str(c).strip().lower() for c in results.columns]
results["domain"] = results["domain"].astype(str).str.capitalize()
results["n_mics"] = results["n_mics"].astype(str).str.upper()
results["config"] = results["config"].astype(str).str.strip().str.upper()

for m in ALL_METRICS:
    if m in results.columns:
        results[m] = pd.to_numeric(results[m], errors="coerce")

results = results.dropna(subset=["drill"])
results["configuration_short"] = (
    results["method"] + " | " + results["domain"] + " | " + results["n_mics"]
)

# WILCOXON SIGNED-RANK TEST WITH EFFECT SIZE |Z| / sqrt(N)
def paired_test(df, value1, value2, metric, comparison_name):
    temp = df[["drill", value1, value2]].dropna()
    x = temp[value1].to_numpy(dtype=float)
    y = temp[value2].to_numpy(dtype=float)

    delta = y - x
    better = int(np.sum(delta > 1e-12))
    worse = int(np.sum(delta < -1e-12))
    equal = int(np.sum(np.abs(delta) <= 1e-12))

    nonzero = delta[np.abs(delta) > 1e-12]
    n_nonzero = len(nonzero)

    if len(delta) == 0:
        statistic, pvalue, effect_size = np.nan, np.nan, np.nan
    elif n_nonzero == 0:
        statistic, pvalue, effect_size = 0.0, 1.0, 0.0
    else:
        statistic, pvalue = wilcoxon(
            x, y, zero_method="wilcox", alternative="two-sided", method="auto"
        )

        # Analytical calculation of W and Z based on ranks
        ranks = rankdata(np.abs(nonzero), method="average")
        W_plus = np.sum(ranks[nonzero > 0])
        W_minus = np.sum(ranks[nonzero < 0])
        W = min(W_plus, W_minus)

        mean_W = n_nonzero * (n_nonzero + 1) / 4
        sd_W = np.sqrt(
            n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24
        )

        if sd_W > 0:
            z = (W - mean_W) / sd_W
            effect_size = abs(z) / np.sqrt(n_nonzero)
        else:
            effect_size = 0.0

    return {
        "comparison": comparison_name,
        "metric": metric,
        "n": len(delta),
        "mean_1": np.mean(x),
        "mean_2": np.mean(y),
        "delta_mean": np.mean(delta),
        "median_1": np.median(x),
        "median_2": np.median(y),
        "delta_median": np.median(delta),
        "better": better,
        "worse": worse,
        "equal": equal,
        "wilcoxon_statistic": statistic,
        "p_value": pvalue,
        "effect_size_r": effect_size,
    }


def significance_label(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def effect_label(r):
    if pd.isna(r):
        return "NA"
    r_abs = abs(r)
    if r_abs < 0.10:
        return "negligible"
    elif r_abs < 0.30:
        return "small"
    elif r_abs < 0.50:
        return "medium"
    return "large"


# GLOBAL FRIEDMAN TEST 

wide = results.pivot_table(
    index="drill",
    columns="configuration_short",
    values=CORE_METRICS,
    aggfunc="mean",
)

friedman_summary = []
for metric in CORE_METRICS:
    metric_data = wide[metric].dropna(axis=0, how="any")
    arrays = [metric_data[c].values for c in metric_data.columns]
    stat, pval = friedmanchisquare(*arrays)
    friedman_summary.append({
        "metric": metric.upper(),
        "chi2": stat,
        "p_value": pval,
        "sig": significance_label(pval),
    })

df_friedman = pd.DataFrame(friedman_summary)
print(df_friedman.to_string(index=False))

df_friedman.to_csv(
    os.path.join(RESULTS_DIR, "friedman_table3.csv"), index=False
)

# POST-HOC ANALYSIS 1: SENSOR DENSITY (1-MIC vs 3-MIC)
mic_tests = []
for method in results["method"].unique():
    for domain in results["domain"].unique():
        subset = results[
            (results["method"] == method) & (results["domain"] == domain)
        ]
        pivot = subset.pivot_table(
            index="drill", columns="n_mics", values=ALL_METRICS, aggfunc="mean"
        )

        for metric in ALL_METRICS:
            if (metric, "1MIC") not in pivot.columns or (
                metric,
                "3MIC",
            ) not in pivot.columns:
                continue
            temp = pd.DataFrame({
                "drill": pivot.index,
                "1MIC": pivot[(metric, "1MIC")],
                "3MIC": pivot[(metric, "3MIC")],
            })
            res = paired_test(
                temp, "1MIC", "3MIC", metric, f"{method} | {domain} | 1MIC -> 3MIC"
            )
            res["method"] = method
            res["domain"] = domain
            mic_tests.append(res)

df_mic = pd.DataFrame(mic_tests)

# POST-HOC ANALYSIS 2: CROSS-SENSOR vs FEATURE FUSION
method_tests = []
for domain in results["domain"].unique():
    for n_mics in results["n_mics"].unique():
        subset = results[
            (results["domain"] == domain) & (results["n_mics"] == n_mics)
        ]
        pivot = subset.pivot_table(
            index="drill", columns="method", values=ALL_METRICS, aggfunc="mean"
        )

        for metric in ALL_METRICS:
            if (metric, "Cross-Sensor") not in pivot.columns or (
                metric,
                "Feature-Level Fusion",
            ) not in pivot.columns:
                continue
            temp = pd.DataFrame({
                "drill": pivot.index,
                "Cross-Sensor": pivot[(metric, "Cross-Sensor")],
                "Feature-Level Fusion": pivot[(metric, "Feature-Level Fusion")],
            })
            res = paired_test(
                temp,
                "Cross-Sensor",
                "Feature-Level Fusion",
                metric,
                f"{domain} | {n_mics} | Cross-Sensor -> Feature Fusion",
            )
            res["domain"] = domain
            res["n_mics"] = n_mics
            method_tests.append(res)

df_methods = pd.DataFrame(method_tests)

# 7. HOLM-BONFERRONI CORRECTION BY METRIC FAMILY
for df_posthoc, name in [
    (df_mic, "1-MIC vs 3-MIC"),
    (df_methods, "Cross-Sensor vs Feature Fusion"),
]:
    df_posthoc["p_value_holm"] = np.nan
    df_posthoc["sig_holm"] = ""

    for metric in df_posthoc["metric"].unique():
        mask = df_posthoc["metric"] == metric
        p_vals = df_posthoc.loc[mask, "p_value"].values
        reject, p_corrected, _, _ = multipletests(p_vals, method="holm")
        df_posthoc.loc[mask, "p_value_holm"] = p_corrected
        df_posthoc.loc[mask, "sig_holm"] = [
            significance_label(p) for p in p_corrected
        ]

    df_posthoc["effect_magnitude"] = [
        effect_label(r) for r in df_posthoc["effect_size_r"]
    ]

    print("\n" + "=" * 80)
    print(f"POST-HOC: {name} (WITH HOLM CORRECTION)")
    print("=" * 80)
    cols_show = [
        "comparison",
        "metric",
        "delta_mean",
        "p_value",
        "p_value_holm",
        "sig_holm",
        "effect_size_r",
        "effect_magnitude",
    ]
    print(df_posthoc[cols_show].round(4).to_string(index=False))

# Export post-hoc statistical reports
df_mic.to_csv(
    os.path.join(RESULTS_DIR, "wilcoxon_holm_1mic_vs_3mic.csv"),
    index=False,
)
df_methods.to_csv(
    os.path.join(RESULTS_DIR, "wilcoxon_holm_cross_vs_fusion.csv"),
    index=False,
)