#!/usr/bin/env python3

"""
Compute correlation between percentiles of inter-sample pairwise similarity and pass@1.

For each problem, the pairwise similarity scores are summarized at each percentile
(0th, 1st, ..., 100th). Then, across all problems, the Pearson and Spearman
correlation between that percentile value and pass@1 is computed.

This reveals which percentile of inter-sample similarity is most predictive of pass@1.

Usage:
    # Using pre-computed scores
    python compute_similarity_pass_correlation.py --work-dir work \
        --scores-json work/inter_sample_similarity.json

    # Compute from scratch
    python compute_similarity_pass_correlation.py --work-dir work

    # With GPU
    python compute_similarity_pass_correlation.py --work-dir work --device cuda

Output:
    - CSV with columns: percentile, bert_pearson_r, bert_pearson_p, bert_spearman_r,
      bert_spearman_p, codebert_pearson_r, codebert_pearson_p, ...
    - PNG plot of correlation vs. percentile
    - JSON with full results

Requirements:
    pip install bert-score transformers torch matplotlib numpy scipy
"""

import argparse
import csv
import json
import os
import re
import sys
import logging
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Pass@1 parsing
# ──────────────────────────────────────────────────────────────────────

def parse_pass_at_1(work_dir: str) -> Dict[str, float]:
    """
    Parse per-problem pass@1 from the composite report.

    Returns:
        Dict mapping problem_id -> pass@1 rate
    """
    report_path = os.path.join(work_dir, "composite_report.txt")
    if not os.path.isfile(report_path):
        logger.warning("composite_report.txt not found at %s", report_path)
        return {}

    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    pass_rates: Dict[str, float] = {}
    current_rate: Optional[float] = None

    for line in text.splitlines():
        rate_match = re.search(r"Pass Rate:\s+([\d.]+)", line)
        if rate_match:
            current_rate = float(rate_match.group(1))
            continue

        if current_rate is not None:
            pid_match = re.search(r"\|\s+(cvdp_\S+)\s+\|", line)
            if pid_match:
                pass_rates[pid_match.group(1)] = current_rate

    logger.info("Parsed pass@1 rates for %d problems", len(pass_rates))
    return pass_rates


# ──────────────────────────────────────────────────────────────────────
# Code loading
# ──────────────────────────────────────────────────────────────────────

def discover_problems(work_dir: str, sample_nums: List[int]) -> Set[str]:
    problems = set()
    for s in sample_nums:
        sample_dir = os.path.join(work_dir, f"sample_{s}")
        if not os.path.isdir(sample_dir):
            continue
        for entry in os.listdir(sample_dir):
            if entry.startswith("cvdp_") and os.path.isdir(os.path.join(sample_dir, entry)):
                problems.add(entry)
    return problems


def load_generated_code(work_dir: str, problem_dir: str, sample: int) -> str:
    harness_dir = os.path.join(work_dir, f"sample_{sample}", problem_dir, "harness")
    if not os.path.isdir(harness_dir):
        return ""

    code_extensions = {".sv", ".v", ".svh", ".vh", ".vhd", ".vhdl"}
    exclude_patterns = {"cocotb_iverilog_dump.v", "sim_build", "rundir"}

    code_files = []
    for root, dirs, files in os.walk(harness_dir):
        dirs[:] = [d for d in dirs if d not in exclude_patterns]
        rel_root = os.path.relpath(root, harness_dir)
        path_parts = rel_root.split(os.sep)
        if len(path_parts) >= 2 and path_parts[1] != "rtl":
            continue
        for fname in sorted(files):
            if fname in exclude_patterns:
                continue
            _, ext = os.path.splitext(fname)
            if ext.lower() in code_extensions:
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        code_files.append((os.path.relpath(fpath, harness_dir), f.read()))
                except Exception as e:
                    logger.warning("Failed to read %s: %s", fpath, e)

    if not code_files:
        return ""
    code_files.sort(key=lambda x: x[0])
    return "\n".join(content for _, content in code_files)


# ──────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────

def compute_bert_scores(
    candidates: List[str], references: List[str],
    device: str = "cpu", batch_size: int = 32,
) -> Tuple[List[float], List[float]]:
    from bert_score import score as bert_score_fn

    logger.info("Computing BERTScore for %d pairs...", len(candidates))
    _, _, F1_bert = bert_score_fn(
        candidates, references, lang="en",
        device=device, batch_size=batch_size, verbose=False,
    )
    logger.info("Computing CodeBERTScore for %d pairs...", len(candidates))
    _, _, F1_code = bert_score_fn(
        candidates, references,
        model_type="microsoft/codebert-base", num_layers=12,
        device=device, batch_size=batch_size, verbose=False,
    )
    return F1_bert.tolist(), F1_code.tolist()


def compute_all_scores(
    work_dir: str, device: str = "cpu", batch_size: int = 32,
) -> Dict[str, Dict[str, List[float]]]:
    sample_nums = sorted(
        int(d.split("_")[1])
        for d in os.listdir(work_dir)
        if d.startswith("sample_") and os.path.isdir(os.path.join(work_dir, d))
    )
    if len(sample_nums) < 2:
        sys.exit("Need >= 2 samples")

    sample_pairs = list(combinations(sample_nums, 2))
    problem_dirs = sorted(discover_problems(work_dir, sample_nums))

    batch_cand, batch_ref = [], []
    scoring_map: List[Tuple[str, Tuple[int, int]]] = []

    for problem_dir in problem_dirs:
        sample_code: Dict[int, str] = {}
        for s in sample_nums:
            code = load_generated_code(work_dir, problem_dir, s)
            if code.strip():
                sample_code[s] = code
        if len(sample_code) < 2:
            continue
        for si, sj in sample_pairs:
            if si in sample_code and sj in sample_code:
                batch_cand.append(sample_code[si])
                batch_ref.append(sample_code[sj])
                scoring_map.append((problem_dir, (si, sj)))

    if not batch_cand:
        sys.exit("No pairs found")

    bert_f1, codebert_f1 = compute_bert_scores(batch_cand, batch_ref, device, batch_size)

    results: Dict[str, Dict[str, List[float]]] = {}
    for idx, (pd, _) in enumerate(scoring_map):
        if pd not in results:
            results[pd] = {"bert_f1": [], "codebert_f1": []}
        results[pd]["bert_f1"].append(bert_f1[idx])
        results[pd]["codebert_f1"].append(codebert_f1[idx])

    return results


def load_scores_from_json(json_path: str) -> Dict[str, Dict[str, List[float]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: Dict[str, Dict[str, List[float]]] = {}
    for problem_id, pairs in data.get("per_pair", {}).items():
        bert_vals, codebert_vals = [], []
        for scores in pairs.values():
            bert_vals.append(scores["bert_f1"])
            codebert_vals.append(scores["codebert_f1"])
        if bert_vals:
            results[problem_id] = {"bert_f1": bert_vals, "codebert_f1": codebert_vals}

    logger.info("Loaded scores for %d problems from %s", len(results), json_path)
    return results


# ──────────────────────────────────────────────────────────────────────
# ID / directory mapping
# ──────────────────────────────────────────────────────────────────────

def problem_id_to_dir(problem_id: str) -> str:
    parts = problem_id.split("_")
    return "_".join(parts[:-1])


def match_pass_rates(
    scores: Dict[str, Dict[str, List[float]]],
    pass_rates: Dict[str, float],
) -> Dict[str, Tuple[Dict[str, List[float]], float]]:
    """
    Match score keys to pass@1 rates.

    Returns:
        Dict mapping key -> (scores_dict, pass@1_rate) for matched problems.
    """
    # Build dir -> list of (problem_id, rate) mapping
    dir_to_rates: Dict[str, List[float]] = {}
    for pid, rate in pass_rates.items():
        d = problem_id_to_dir(pid)
        dir_to_rates.setdefault(d, []).append(rate)

    matched = {}
    for key, sdict in scores.items():
        if key in pass_rates:
            matched[key] = (sdict, pass_rates[key])
        elif key in dir_to_rates:
            avg_rate = sum(dir_to_rates[key]) / len(dir_to_rates[key])
            matched[key] = (sdict, avg_rate)
        else:
            d = problem_id_to_dir(key)
            if d in dir_to_rates:
                avg_rate = sum(dir_to_rates[d]) / len(dir_to_rates[d])
                matched[key] = (sdict, avg_rate)

    logger.info("Matched %d / %d problems with pass@1 data", len(matched), len(scores))
    return matched


# ──────────────────────────────────────────────────────────────────────
# Correlation computation
# ──────────────────────────────────────────────────────────────────────

def compute_correlations(
    matched: Dict[str, Tuple[Dict[str, List[float]], float]],
    percentiles: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For each percentile, compute Pearson and Spearman correlation between
    that percentile of the inter-sample scores and pass@1.

    Returns:
        Dict with keys "bert_f1" and "codebert_f1", each containing:
            "pearson_r", "pearson_p", "spearman_r", "spearman_p"
            as arrays of shape (len(percentiles),)
    """
    # Extract pass@1 vector (same for both metrics)
    keys = sorted(matched.keys())
    pass_vec = np.array([matched[k][1] for k in keys])

    results = {}

    for metric in ["bert_f1", "codebert_f1"]:
        # Build matrix: (n_problems, n_percentiles)
        pct_matrix = np.zeros((len(keys), len(percentiles)))
        for i, k in enumerate(keys):
            vals = matched[k][0][metric]
            pct_matrix[i, :] = np.percentile(vals, percentiles)

        pearson_r = np.zeros(len(percentiles))
        pearson_p = np.zeros(len(percentiles))
        spearman_r = np.zeros(len(percentiles))
        spearman_p = np.zeros(len(percentiles))

        for j in range(len(percentiles)):
            col = pct_matrix[:, j]

            # Pearson
            if np.std(col) > 0 and np.std(pass_vec) > 0:
                r, p = stats.pearsonr(col, pass_vec)
                pearson_r[j] = r
                pearson_p[j] = p
            else:
                pearson_r[j] = 0.0
                pearson_p[j] = 1.0

            # Spearman
            if np.std(col) > 0 and np.std(pass_vec) > 0:
                r, p = stats.spearmanr(col, pass_vec)
                spearman_r[j] = r
                spearman_p[j] = p
            else:
                spearman_r[j] = 0.0
                spearman_p[j] = 1.0

        results[metric] = {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        }

    return results


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_correlations(
    corr_results: Dict[str, Dict[str, np.ndarray]],
    percentiles: np.ndarray,
    output_path: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metric_labels = {
        "bert_f1": "BERTScore F1",
        "codebert_f1": "CodeBERTScore F1",
    }

    for col, metric in enumerate(["bert_f1", "codebert_f1"]):
        data = corr_results[metric]
        label = metric_labels[metric]

        # Top row: Correlation coefficient
        ax_r = axes[0][col]
        ax_r.plot(percentiles, data["pearson_r"], color="steelblue", linewidth=2, label="Pearson r")
        ax_r.plot(percentiles, data["spearman_r"], color="darkorange", linewidth=2, label="Spearman \u03c1")
        ax_r.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_r.set_ylabel("Correlation with pass@1", fontsize=11)
        ax_r.set_title(f"{label}\nCorrelation vs. Percentile", fontsize=13, fontweight="bold")
        ax_r.legend(loc="best", fontsize=10)
        ax_r.set_xlim(0, 100)
        ax_r.grid(True, alpha=0.3)

        # Annotate peak
        best_pearson_idx = np.argmax(np.abs(data["pearson_r"]))
        best_spearman_idx = np.argmax(np.abs(data["spearman_r"]))
        ax_r.annotate(
            f"peak r={data['pearson_r'][best_pearson_idx]:.3f}\n@ p{percentiles[best_pearson_idx]:.0f}",
            xy=(percentiles[best_pearson_idx], data["pearson_r"][best_pearson_idx]),
            fontsize=8, color="steelblue", fontweight="bold",
            textcoords="offset points", xytext=(10, -15),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8),
        )
        ax_r.annotate(
            f"peak \u03c1={data['spearman_r'][best_spearman_idx]:.3f}\n@ p{percentiles[best_spearman_idx]:.0f}",
            xy=(percentiles[best_spearman_idx], data["spearman_r"][best_spearman_idx]),
            fontsize=8, color="darkorange", fontweight="bold",
            textcoords="offset points", xytext=(10, 10),
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8),
        )

        # Bottom row: p-value (log scale)
        ax_p = axes[1][col]
        ax_p.semilogy(percentiles, data["pearson_p"], color="steelblue", linewidth=2, label="Pearson p-value")
        ax_p.semilogy(percentiles, data["spearman_p"], color="darkorange", linewidth=2, label="Spearman p-value")
        ax_p.axhline(0.05, color="red", linestyle="--", linewidth=1, alpha=0.7, label="p = 0.05")
        ax_p.axhline(0.01, color="darkred", linestyle=":", linewidth=1, alpha=0.7, label="p = 0.01")
        ax_p.set_xlabel("Percentile", fontsize=11)
        ax_p.set_ylabel("p-value (log scale)", fontsize=11)
        ax_p.set_title(f"{label}\np-value vs. Percentile", fontsize=13, fontweight="bold")
        ax_p.legend(loc="best", fontsize=9)
        ax_p.set_xlim(0, 100)
        ax_p.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", output_path)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute correlation between percentiles of inter-sample "
                    "similarity and pass@1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_similarity_pass_correlation.py --work-dir work \\
      --scores-json work/inter_sample_similarity.json
  python compute_similarity_pass_correlation.py --work-dir work --device cuda
  python compute_similarity_pass_correlation.py --work-dir work -o results/corr
        """,
    )
    parser.add_argument("--work-dir", type=str, default="work")
    parser.add_argument("--scores-json", type=str, default=None,
                       help="Pre-computed inter-sample similarity JSON")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Output prefix (default: <work-dir>/similarity_pass_correlation)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--percentile-step", type=int, default=1,
                       help="Step between percentiles (default: 1, i.e. 0,1,2,...,100)")

    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        sys.exit(f"Error: Work directory not found: {args.work_dir}")

    output_prefix = args.output or os.path.join(args.work_dir, "similarity_pass_correlation")
    output_csv = output_prefix + ".csv"
    output_json = output_prefix + ".json"
    output_png = output_prefix + ".png"

    # Parse pass@1
    pass_rates = parse_pass_at_1(args.work_dir)
    if not pass_rates:
        sys.exit("Error: Could not parse pass@1 from composite_report.txt")

    # Get scores
    if args.scores_json:
        scores = load_scores_from_json(args.scores_json)
    else:
        scores = compute_all_scores(args.work_dir, args.device, args.batch_size)

    if not scores:
        sys.exit("Error: No similarity scores found.")

    # Match scores with pass rates
    matched = match_pass_rates(scores, pass_rates)
    if len(matched) < 3:
        sys.exit(f"Error: Only {len(matched)} problems matched. Need at least 3 for correlation.")

    # Define percentiles
    percentiles = np.arange(0, 101, args.percentile_step, dtype=float)

    # Compute correlations
    logger.info("Computing correlations for %d percentiles across %d problems...",
                len(percentiles), len(matched))
    corr_results = compute_correlations(matched, percentiles)

    # ── Write CSV ──
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "percentile",
            "bert_pearson_r", "bert_pearson_p", "bert_spearman_r", "bert_spearman_p",
            "codebert_pearson_r", "codebert_pearson_p", "codebert_spearman_r", "codebert_spearman_p",
        ])
        for i, pct in enumerate(percentiles):
            writer.writerow([
                int(pct),
                round(corr_results["bert_f1"]["pearson_r"][i], 6),
                round(corr_results["bert_f1"]["pearson_p"][i], 6),
                round(corr_results["bert_f1"]["spearman_r"][i], 6),
                round(corr_results["bert_f1"]["spearman_p"][i], 6),
                round(corr_results["codebert_f1"]["pearson_r"][i], 6),
                round(corr_results["codebert_f1"]["pearson_p"][i], 6),
                round(corr_results["codebert_f1"]["spearman_r"][i], 6),
                round(corr_results["codebert_f1"]["spearman_p"][i], 6),
            ])
    logger.info("Saved CSV to %s", output_csv)

    # ── Write JSON ──
    def _best(arr):
        idx = int(np.argmax(np.abs(arr)))
        return {"percentile": int(percentiles[idx]), "value": round(float(arr[idx]), 6)}

    json_output = {
        "num_problems": len(matched),
        "percentile_step": args.percentile_step,
        "bert_f1": {
            "best_pearson": _best(corr_results["bert_f1"]["pearson_r"]),
            "best_spearman": _best(corr_results["bert_f1"]["spearman_r"]),
        },
        "codebert_f1": {
            "best_pearson": _best(corr_results["codebert_f1"]["pearson_r"]),
            "best_spearman": _best(corr_results["codebert_f1"]["spearman_r"]),
        },
        "per_percentile": [],
    }
    for i, pct in enumerate(percentiles):
        json_output["per_percentile"].append({
            "percentile": int(pct),
            "bert_f1": {
                "pearson_r": round(float(corr_results["bert_f1"]["pearson_r"][i]), 6),
                "pearson_p": round(float(corr_results["bert_f1"]["pearson_p"][i]), 6),
                "spearman_r": round(float(corr_results["bert_f1"]["spearman_r"][i]), 6),
                "spearman_p": round(float(corr_results["bert_f1"]["spearman_p"][i]), 6),
            },
            "codebert_f1": {
                "pearson_r": round(float(corr_results["codebert_f1"]["pearson_r"][i]), 6),
                "pearson_p": round(float(corr_results["codebert_f1"]["pearson_p"][i]), 6),
                "spearman_r": round(float(corr_results["codebert_f1"]["spearman_r"][i]), 6),
                "spearman_p": round(float(corr_results["codebert_f1"]["spearman_p"][i]), 6),
            },
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    logger.info("Saved JSON to %s", output_json)

    # ── Plot ──
    plot_correlations(corr_results, percentiles, output_png)

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("SIMILARITY-PERCENTILE vs. PASS@1 CORRELATION")
    print("=" * 70)
    print(f"Problems:        {len(matched)}")
    print(f"Percentile step: {args.percentile_step}")
    print("-" * 70)

    for metric, label in [("bert_f1", "BERTScore F1"), ("codebert_f1", "CodeBERTScore F1")]:
        bp = json_output[metric]["best_pearson"]
        bs = json_output[metric]["best_spearman"]
        print(f"\n  {label}:")
        print(f"    Best Pearson r:  {bp['value']:+.4f}  @ percentile {bp['percentile']}")
        print(f"    Best Spearman ρ: {bs['value']:+.4f}  @ percentile {bs['percentile']}")

    print("\n" + "=" * 70)
    print(f"Results saved to:\n  CSV:  {output_csv}\n  JSON: {output_json}\n  Plot: {output_png}")


if __name__ == "__main__":
    main()
