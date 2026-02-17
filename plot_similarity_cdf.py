#!/usr/bin/env python3

"""
Plot CDF of inter-sample BERTScore and CodeBERTScore, colored by pass@1.

For each problem, plots a CDF line over all its pairwise inter-sample similarity
scores. Problems with pass@1 > 0.5 are colored blue; pass@1 <= 0.5 are red.

Usage:
    # Compute scores from scratch and plot
    python plot_similarity_cdf.py --work-dir work

    # Use pre-computed similarity JSON (from compute_code_similarity.py)
    python plot_similarity_cdf.py --work-dir work --scores-json work/inter_sample_similarity.json

    # Use GPU for faster scoring
    python plot_similarity_cdf.py --work-dir work --device cuda

    # Custom output path
    python plot_similarity_cdf.py --work-dir work -o my_cdf_plot.png

Requirements:
    pip install bert-score transformers torch matplotlib numpy
"""

import argparse
import json
import os
import re
import sys
import logging
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Pass@1 parsing
# ──────────────────────────────────────────────────────────────────────

def parse_pass_at_1(work_dir: str) -> Dict[str, float]:
    """
    Parse per-problem pass@1 from the composite report.

    The composite_report.txt groups problems under pass-count headers like:
        | Pass Count: 0/5 |
        | Pass Rate: 0.0000 |
        ...
        | cvdp_copilot_some_problem_0001 | ...

    Returns:
        Dict mapping problem_id -> pass@1 rate (e.g. 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
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
        # Match "Pass Rate: 0.4000" lines
        rate_match = re.search(r"Pass Rate:\s+([\d.]+)", line)
        if rate_match:
            current_rate = float(rate_match.group(1))
            continue

        # Match problem ID lines like "| cvdp_copilot_xxx_0001 |"
        if current_rate is not None:
            pid_match = re.search(r"\|\s+(cvdp_\S+)\s+\|", line)
            if pid_match:
                problem_id = pid_match.group(1)
                pass_rates[problem_id] = current_rate

    logger.info("Parsed pass@1 rates for %d problems", len(pass_rates))
    return pass_rates


# ──────────────────────────────────────────────────────────────────────
# Code loading (same logic as compute_code_similarity.py)
# ──────────────────────────────────────────────────────────────────────

def discover_problems(work_dir: str, sample_nums: List[int]) -> Set[str]:
    """Discover problem directory names across samples."""
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
    """Load and concatenate all RTL files for a problem+sample."""
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

def compute_scores(
    candidates: List[str],
    references: List[str],
    device: str = "cpu",
    batch_size: int = 32,
) -> Tuple[List[float], List[float]]:
    """Compute BERTScore and CodeBERTScore F1."""
    from bert_score import score as bert_score_fn

    logger.info("Computing BERTScore (roberta-large) for %d pairs...", len(candidates))
    _, _, F1_bert = bert_score_fn(
        candidates, references, lang="en",
        device=device, batch_size=batch_size, verbose=False,
    )

    logger.info("Computing CodeBERTScore (microsoft/codebert-base) for %d pairs...", len(candidates))
    _, _, F1_code = bert_score_fn(
        candidates, references,
        model_type="microsoft/codebert-base", num_layers=12,
        device=device, batch_size=batch_size, verbose=False,
    )

    return F1_bert.tolist(), F1_code.tolist()


def compute_all_inter_sample_scores(
    work_dir: str, device: str = "cpu", batch_size: int = 32,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute inter-sample similarity for all problems.

    Returns:
        Dict mapping problem_dir -> {"bert_f1": [...], "codebert_f1": [...]}
    """
    sample_nums = sorted(
        int(d.split("_")[1])
        for d in os.listdir(work_dir)
        if d.startswith("sample_") and os.path.isdir(os.path.join(work_dir, d))
    )
    if len(sample_nums) < 2:
        logger.error("Need >= 2 samples, found %d", len(sample_nums))
        sys.exit(1)

    sample_pairs = list(combinations(sample_nums, 2))
    problem_dirs = sorted(discover_problems(work_dir, sample_nums))
    logger.info("Discovered %d problems, %d sample pairs", len(problem_dirs), len(sample_pairs))

    # Build batch
    batch_cand: List[str] = []
    batch_ref: List[str] = []
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
        logger.error("No inter-sample pairs found")
        sys.exit(1)

    logger.info("Scoring %d pairs...", len(batch_cand))
    bert_f1, codebert_f1 = compute_scores(batch_cand, batch_ref, device, batch_size)

    # Organize per problem
    results: Dict[str, Dict[str, List[float]]] = {}
    for idx, (problem_dir, _pair) in enumerate(scoring_map):
        if problem_dir not in results:
            results[problem_dir] = {"bert_f1": [], "codebert_f1": []}
        results[problem_dir]["bert_f1"].append(bert_f1[idx])
        results[problem_dir]["codebert_f1"].append(codebert_f1[idx])

    return results


def load_scores_from_json(json_path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load pre-computed scores from the JSON output of compute_code_similarity.py.

    Returns:
        Dict mapping problem_dir -> {"bert_f1": [...], "codebert_f1": [...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: Dict[str, Dict[str, List[float]]] = {}
    for problem_id, pairs in data.get("per_pair", {}).items():
        bert_vals = []
        codebert_vals = []
        for _pair_key, scores in pairs.items():
            bert_vals.append(scores["bert_f1"])
            codebert_vals.append(scores["codebert_f1"])
        if bert_vals:
            results[problem_id] = {
                "bert_f1": bert_vals,
                "codebert_f1": codebert_vals,
            }

    logger.info("Loaded scores for %d problems from %s", len(results), json_path)
    return results


# ──────────────────────────────────────────────────────────────────────
# Mapping between problem IDs and directory names
# ──────────────────────────────────────────────────────────────────────

def problem_id_to_dir(problem_id: str) -> str:
    """
    Convert a problem ID like 'cvdp_copilot_64b66b_decoder_0001'
    to its directory name 'cvdp_copilot_64b66b_decoder'.
    """
    parts = problem_id.split("_")
    return "_".join(parts[:-1])


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_cdf(
    scores_per_problem: Dict[str, Dict[str, List[float]]],
    pass_rates: Dict[str, float],
    output_path: str,
    threshold: float = 0.5,
):
    """
    Plot CDF of BERTScore and CodeBERTScore, one line per problem,
    colored by pass@1 threshold.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build mapping: problem_dir -> pass@1
    dir_to_pass = {}
    for pid, rate in pass_rates.items():
        d = problem_id_to_dir(pid)
        # A problem can have multiple issue numbers under the same dir
        # Use max pass rate if multiple (shouldn't happen, but be safe)
        if d not in dir_to_pass:
            dir_to_pass[d] = {}
        dir_to_pass[d][pid] = rate

    # For each problem_dir in scores, determine pass@1
    # Scores keys might be dir names (from compute) or full IDs (from JSON)
    problem_pass: Dict[str, float] = {}
    for key in scores_per_problem:
        # Try direct match in pass_rates (full problem ID)
        if key in pass_rates:
            problem_pass[key] = pass_rates[key]
        # Try as directory name
        elif key in dir_to_pass:
            # Average across issues under this dir
            rates = list(dir_to_pass[key].values())
            problem_pass[key] = sum(rates) / len(rates)
        else:
            # Try stripping to dir name
            d = problem_id_to_dir(key)
            if d in dir_to_pass:
                rates = list(dir_to_pass[d].values())
                problem_pass[key] = sum(rates) / len(rates)
            else:
                problem_pass[key] = 0.0  # Unknown defaults to failing

    # Split into high/low pass groups
    high_pass = {k: v for k, v in scores_per_problem.items() if problem_pass.get(k, 0) > threshold}
    low_pass = {k: v for k, v in scores_per_problem.items() if problem_pass.get(k, 0) <= threshold}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, metric, title in [
        (axes[0], "bert_f1", "BERTScore"),
        (axes[1], "codebert_f1", "CodeBERTScore"),
    ]:
        # Collect all values for this metric to determine x-axis range
        all_vals = []
        for scores_dict in scores_per_problem.values():
            all_vals.extend(scores_dict[metric])

        # Set x-axis to span the data with some padding
        if all_vals:
            data_min = min(all_vals)
            data_max = max(all_vals)
            x_range = data_max - data_min
            padding = max(x_range * 0.1, 0.02)  # At least 0.02 padding
            x_lo = max(0.0, np.floor((data_min - padding) * 20) / 20)  # Round down to nearest 0.05
            x_hi = min(1.0, np.ceil((data_max + padding) * 20) / 20)   # Round up to nearest 0.05
        else:
            x_lo, x_hi = 0.0, 1.0

        # Plot low-pass (red) first, then high-pass (blue) on top
        for group, color, label_prefix in [
            (low_pass, "red", f"pass@1 \u2264 {threshold}"),
            (high_pass, "blue", f"pass@1 > {threshold}"),
        ]:
            n_problems = len(group)
            label_done = False
            for _pid, scores_dict in sorted(group.items()):
                vals = sorted(scores_dict[metric])
                n = len(vals)
                if n == 0:
                    continue
                # CDF: y goes from 0 to 1
                cdf_y = np.arange(1, n + 1) / n
                label = f"{label_prefix} (n={n_problems})" if not label_done else None
                ax.plot(
                    vals, cdf_y,
                    color=color, alpha=0.35, linewidth=1.0,
                    label=label,
                )
                label_done = True

        ax.set_xlabel(f"{title} F1", fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontsize=12)
        ax.set_title(
            f"CDF of {title} by Problem\n"
            f"(Blue: pass@1 > {threshold}, Red: pass@1 \u2264 {threshold})",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", output_path)
    print(f"\nPlot saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot CDF of inter-sample BERTScore/CodeBERTScore colored by pass@1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_similarity_cdf.py --work-dir work
  python plot_similarity_cdf.py --work-dir work --scores-json work/inter_sample_similarity.json
  python plot_similarity_cdf.py --work-dir work --device cuda -o my_plot.png
        """,
    )
    parser.add_argument(
        "--work-dir", type=str, default="work",
        help="Root work directory (default: work)",
    )
    parser.add_argument(
        "--scores-json", type=str, default=None,
        help="Pre-computed similarity JSON (from compute_code_similarity.py). "
             "If omitted, scores are computed from scratch.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output plot path (default: <work-dir>/similarity_cdf.png)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="pass@1 threshold for coloring (default: 0.5)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for scoring (default: cpu). Ignored if --scores-json is provided.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for scoring (default: 32). Ignored if --scores-json is provided.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        print(f"Error: Work directory not found: {args.work_dir}")
        sys.exit(1)

    output_path = args.output or os.path.join(args.work_dir, "similarity_cdf.png")

    # Parse pass@1
    pass_rates = parse_pass_at_1(args.work_dir)
    if not pass_rates:
        print("Error: Could not parse pass@1 from composite_report.txt")
        sys.exit(1)

    # Get scores
    if args.scores_json:
        scores = load_scores_from_json(args.scores_json)
    else:
        scores = compute_all_inter_sample_scores(
            args.work_dir, device=args.device, batch_size=args.batch_size,
        )
        # Also save the computed scores for reuse
        save_path = os.path.join(args.work_dir, "inter_sample_similarity.json")
        # Build minimal JSON structure expected by load_scores_from_json
        per_pair_json = {}
        for pid, sdict in scores.items():
            per_pair_json[pid] = {
                f"pair_{i}": {"bert_f1": b, "codebert_f1": c}
                for i, (b, c) in enumerate(zip(sdict["bert_f1"], sdict["codebert_f1"]))
            }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"per_pair": per_pair_json}, f, indent=2)
        logger.info("Saved computed scores to %s", save_path)

    if not scores:
        print("Error: No similarity scores found.")
        sys.exit(1)

    # Plot
    plot_cdf(scores, pass_rates, output_path, threshold=args.threshold)


if __name__ == "__main__":
    main()
