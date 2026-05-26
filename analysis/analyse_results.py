import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import config



PRIMARY_METRIC = "total_time_min"
SECONDARY_METRICS = ["mean_lap_time_s", "off_track_count", "fuel_remaining_kg"]
ALL_METRICS_FOR_DESC = [
    "total_time_min", "mean_lap_time_s", "best_lap_s", "worst_lap_s",
    "lap_time_std_s", "pit_stops", "n_compounds",
    "off_track_count", "fuel_remaining_kg", "final_tyre_wear_pct",
]

AGENT_COLORS = {
    "RuleBased": "#5F9EA0",
    "Utility":   "#DAA520",
    "RL":        "#6BD8C8",
}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:

    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    diff_mean = float(np.mean(a) - np.mean(b))
    pooled_var = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0
    if pooled_var < 1e-12:
        return float("nan") if abs(diff_mean) > 1e-9 else 0.0
    return diff_mean / float(np.sqrt(pooled_var))


def rank_biserial(diff: np.ndarray) -> float:

    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    abs_ranks = stats.rankdata(np.abs(diff))
    pos = abs_ranks[diff > 0].sum()
    neg = abs_ranks[diff < 0].sum()
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def effect_label(d: float) -> str:

    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "n/a (zero variance)"
    a = abs(d)
    if a < 0.2:  return "negligible"
    if a < 0.5:  return "small"
    if a < 0.8:  return "medium"
    return "large"


def pairwise_wilcoxon(df: pd.DataFrame, scenario: str, agent_a: str, agent_b: str,
                      metric: str) -> dict:

    a = df[(df["scenario"] == scenario) & (df["agent"] == agent_a)] \
        .sort_values("seed")[metric].values
    b = df[(df["scenario"] == scenario) & (df["agent"] == agent_b)] \
        .sort_values("seed")[metric].values
    if len(a) != len(b) or len(a) < 2:
        return None
    diff = a - b
    # Wilcoxon is undefined if all differences are zero
    if np.all(diff == 0):
        return {
            "agent_a": agent_a, "agent_b": agent_b, "scenario": scenario,
            "n": len(a), "median_a": np.median(a), "median_b": np.median(b),
            "mean_diff": 0.0, "test": "n/a", "stat": None, "p_value": None,
            "rank_biserial": 0.0, "cohens_d": 0.0,
            "note": "zero variance — deterministic scenario, no test possible",
        }
    try:

        stat, p = stats.wilcoxon(a, b, alternative="two-sided",
                                  zero_method="wilcox")
    except ValueError as e:
        return {
            "agent_a": agent_a, "agent_b": agent_b, "scenario": scenario,
            "n": len(a), "median_a": np.median(a), "median_b": np.median(b),
            "mean_diff": float(np.mean(diff)),
            "test": "wilcoxon-failed", "stat": None, "p_value": None,
            "rank_biserial": rank_biserial(diff),
            "cohens_d": cohens_d(a, b),
            "note": str(e),
        }
    return {
        "agent_a": agent_a, "agent_b": agent_b, "scenario": scenario,
        "n": len(a), "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "mean_diff": float(np.mean(diff)),
        "test": "wilcoxon signed-rank",
        "stat": float(stat), "p_value": float(p),
        "rank_biserial": rank_biserial(diff),
        "cohens_d": cohens_d(a, b),
        "note": "",
    }



# Reports


def descriptive_table(df: pd.DataFrame, metric: str) -> str:

    g = df.groupby(["scenario", "agent"])[metric].agg(
        N="count", mean="mean", std="std", min="min",
        median="median", max="max"
    ).round(3)
    return g.to_string()


def dnf_summary(df: pd.DataFrame) -> str:

    total = df.groupby(["scenario", "agent"]).size().rename("N")
    finished = df.groupby(["scenario", "agent"])["finished"].sum().rename("finished")
    dnfs = df.groupby(["scenario", "agent"])["dnf"].sum().rename("dnfs")
    out = pd.concat([total, finished, dnfs], axis=1)
    out["finish_rate"] = (out["finished"] / out["N"]).round(3)
    return out.to_string()


def run_pairwise_tests(df: pd.DataFrame, metric: str,
                       alpha: float = 0.05) -> Tuple[pd.DataFrame, float]:

    scenarios = sorted(df["scenario"].unique())
    agents = sorted(df["agent"].unique())
    pairs = [(agents[i], agents[j])
             for i in range(len(agents)) for j in range(i+1, len(agents))]

    rows = []
    for scenario in scenarios:
        for a, b in pairs:
            r = pairwise_wilcoxon(df, scenario, a, b, metric)
            if r is not None:
                rows.append(r)
    out = pd.DataFrame(rows)


    valid = out[out["test"] == "wilcoxon signed-rank"].copy()
    n_tests = len(valid)
    corrected_alpha = alpha / max(1, n_tests)
    if "p_value" in out.columns:
        out["significant_uncorrected"] = out["p_value"].apply(
            lambda p: (p is not None and p < alpha))
        out["significant_bonferroni"] = out["p_value"].apply(
            lambda p: (p is not None and p < corrected_alpha))
    return out, corrected_alpha


# Plots


def setup_style():
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["font.family"] = "DejaVu Sans"


def boxplot_metric(df, metric, ylabel, title, outpath):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(data=df, x="scenario", y=metric, hue="agent",
                palette=AGENT_COLORS, ax=ax, showfliers=True, width=0.65)
    sns.stripplot(data=df, x="scenario", y=metric, hue="agent",
                  palette=AGENT_COLORS, ax=ax, dodge=True, size=2.5,
                  alpha=0.35, legend=False)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Weather Scenario", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(title="Agent", loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def bar_plot_mean(df, metric, ylabel, title, outpath):
    g = df.groupby(["scenario", "agent"])[metric].agg(["mean", "std"]).reset_index()
    scenarios = sorted(df["scenario"].unique())
    agents = sorted(df["agent"].unique())
    x = np.arange(len(scenarios))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, a in enumerate(agents):
        sub = g[g["agent"] == a].set_index("scenario").loc[scenarios]
        ax.bar(x + (i - 1) * width, sub["mean"], width,
               yerr=sub["std"], capsize=4,
               label=a, color=AGENT_COLORS.get(a, "#999"),
               edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(scenarios)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Weather Scenario", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(title="Agent", loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def compound_usage_heatmap(df, outpath):
    df = df.copy()
    df["compounds_used"] = df["compounds_used"].fillna("")
    rows = []
    for (scenario, agent), grp in df.groupby(["scenario", "agent"]):
        counter = {}
        for cs in grp["compounds_used"]:
            for c in cs.split("+"):
                if c:
                    counter[c] = counter.get(c, 0) + 1
        for c, n in counter.items():
            rows.append({"scenario": scenario, "agent": agent,
                         "compound": c, "n_runs": n})
    mat = pd.DataFrame(rows)
    if mat.empty:
        return
    pivot = mat.pivot_table(index=["scenario", "agent"], columns="compound",
                            values="n_runs", fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "races using compound"})
    ax.set_title("Tyre Compound Usage by (Scenario, Agent)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def pvalue_heatmap(test_results, outpath, corrected_alpha):
    valid = test_results[test_results["test"] == "wilcoxon signed-rank"].copy()
    if valid.empty:
        return
    valid["pair"] = valid["agent_a"] + " vs " + valid["agent_b"]
    pivot = valid.pivot(index="pair", columns="scenario", values="p_value")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn_r",
                vmin=0, vmax=0.1, cbar_kws={"label": "p-value"}, ax=ax)
    ax.set_title(
        f"Wilcoxon Signed-Rank p-values (Bonferroni α = {corrected_alpha:.4f})",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def effect_size_heatmap(test_results, outpath):
    valid = test_results[test_results["test"] == "wilcoxon signed-rank"].copy()
    if valid.empty:
        return
    valid["pair"] = valid["agent_a"] + " vs " + valid["agent_b"]
    pivot = valid.pivot(index="pair", columns="scenario", values="rank_biserial")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r",
                vmin=-1, vmax=1, center=0,
                cbar_kws={"label": "rank-biserial effect size r"}, ax=ax)
    ax.set_title("Effect Size (Rank-Biserial Correlation)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


# Main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to experiments CSV")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", default="results",
                        help="Output directory (default: results)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"ERROR: {args.csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows from {args.csv_path}")
    print(f"Agents:    {sorted(df['agent'].unique())}")
    print(f"Scenarios: {sorted(df['scenario'].unique())}")
    print()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.out_dir, f"analysis_{ts}.txt")
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    lines: List[str] = []
    def w(s=""):
        lines.append(s)
        print(s)

    w("=" * 90)
    w(" DIA F1 — STATISTICAL ANALYSIS OF AGENT COMPARISON")
    w(f" Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f" Source CSV:  {args.csv_path}")
    w(f" N runs:      {len(df)}")
    w("=" * 90)
    w()


    w("-" * 90)
    w(" 1. FINISH / DNF SUMMARY")
    w("-" * 90)
    w(dnf_summary(df))
    w()

    if df["dnf"].sum() > 0:
        w("DNF reasons breakdown:")
        dnf_df = df[df["dnf"]].groupby(["agent", "scenario",
                                         "dnf_reason"]).size().rename("n")
        w(dnf_df.to_string())
        w()

    # Descriptive
    w("-" * 90)
    w(f" 2. DESCRIPTIVE STATISTICS — PRIMARY METRIC: {PRIMARY_METRIC}")
    w("-" * 90)
    w(descriptive_table(df, PRIMARY_METRIC))
    w()

    for m in SECONDARY_METRICS:
        w("-" * 90)
        w(f" 2.{SECONDARY_METRICS.index(m)+2} DESCRIPTIVE — {m}")
        w("-" * 90)
        w(descriptive_table(df, m))
        w()

    # Statistical tests
    w("=" * 90)
    w(f" 3. PAIRWISE STATISTICAL TESTS — primary metric: {PRIMARY_METRIC}")
    w("=" * 90)
    test_results, corrected_alpha = run_pairwise_tests(df, PRIMARY_METRIC)
    n_valid = (test_results["test"] == "wilcoxon signed-rank").sum()
    n_skipped = (test_results["test"] == "n/a").sum()
    w(f" Valid tests:    {n_valid}")
    w(f" Skipped (zero variance):  {n_skipped}")
    w(f" Bonferroni-corrected α:   {corrected_alpha:.4f}  (= 0.05 / {n_valid})")
    w()
    cols = ["scenario", "agent_a", "agent_b", "n",
            "median_a", "median_b", "mean_diff",
            "p_value", "rank_biserial", "cohens_d",
            "significant_uncorrected", "significant_bonferroni"]
    display = test_results[cols].copy() if all(c in test_results.columns for c in cols) else test_results
    if "p_value" in display.columns:
        display["p_value"] = display["p_value"].apply(
            lambda v: "" if v is None or pd.isna(v) else f"{v:.4f}")
    if "median_a" in display.columns:
        display["median_a"] = display["median_a"].round(3)
        display["median_b"] = display["median_b"].round(3)
        display["mean_diff"] = display["mean_diff"].round(3)
        display["rank_biserial"] = display["rank_biserial"].round(3)
        display["cohens_d"] = display["cohens_d"].apply(
            lambda v: "n/a" if (v is None or pd.isna(v)) else f"{v:.3f}")
    w(display.to_string(index=False))
    w()

    # Interpretation summary
    w("-" * 90)
    w(" 4. INTERPRETATION SUMMARY")
    w("-" * 90)
    valid = test_results[test_results["test"] == "wilcoxon signed-rank"]
    for _, row in valid.iterrows():
        bonf_marker = "**" if row["significant_bonferroni"] else \
                      ("*"  if row["significant_uncorrected"] else " ")
        better = row["agent_b"] if row["mean_diff"] > 0 else row["agent_a"]
        delta = abs(row["mean_diff"])
        d = row["cohens_d"]
        d_str = "n/a" if (d is None or pd.isna(d)) else f"{abs(d):.2f}"
        eff = effect_label(d)
        w(f" {bonf_marker} {row['scenario']:10s}  "
          f"{row['agent_a']:10s} vs {row['agent_b']:10s} | "
          f"p={row['p_value']:.4f}  |d|={d_str} ({eff})  "
          f"|  {better} faster by {delta:.2f} min")
    w()
    w(" ** = significant after Bonferroni correction (α = "
      f"{corrected_alpha:.4f})")
    w(" *  = significant uncorrected (p < 0.05) but not after correction")
    w()

    #  Save report
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to: {report_path}")

    # Plots
    if not args.no_plots:
        print("\nGenerating plots…")
        setup_style()

        boxplot_metric(df, "total_time_min",
                       "Total race time (minutes)",
                       "Total Race Time by Agent and Scenario",
                       os.path.join(plots_dir, "boxplot_total_time.png"))
        boxplot_metric(df, "mean_lap_time_s",
                       "Mean lap time (seconds)",
                       "Mean Lap Time by Agent and Scenario",
                       os.path.join(plots_dir, "boxplot_mean_lap.png"))
        boxplot_metric(df, "off_track_count",
                       "Off-track excursions",
                       "Off-Track Count by Agent and Scenario",
                       os.path.join(plots_dir, "boxplot_offtracks.png"))
        boxplot_metric(df, "fuel_remaining_kg",
                       "Fuel remaining at finish (kg)",
                       "Fuel Margin by Agent and Scenario",
                       os.path.join(plots_dir, "boxplot_fuel.png"))

        bar_plot_mean(df, "total_time_min",
                      "Mean total time (minutes) ± SD",
                      "Mean Race Time with Variability",
                      os.path.join(plots_dir, "bar_mean_total_time.png"))

        compound_usage_heatmap(df,
                               os.path.join(plots_dir, "heatmap_compounds.png"))

        pvalue_heatmap(test_results,
                       os.path.join(plots_dir, "heatmap_pvalues.png"),
                       corrected_alpha)
        effect_size_heatmap(test_results,
                            os.path.join(plots_dir, "heatmap_effect_sizes.png"))

        print(f"Plots saved to: {plots_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()