#!/usr/bin/env python3

"""
Compute inter-sample pair-wise BERTScore and CodeBERTScore.

For each problem, compares generated code across all sample pairs (sample_i vs sample_j).
This measures how similar/diverse the model's outputs are across different runs.

Output CSV format (one row per problem, one column per sample pair):
    problem_id, bert_f1_(1,2), bert_f1_(1,3), ..., codebert_f1_(1,2), codebert_f1_(1,3), ...

Usage:
    python compute_code_similarity.py --work-dir work
    python compute_code_similarity.py --work-dir work --device cuda
    python compute_code_similarity.py --work-dir work -o my_results

Requirements:
    pip install bert-score transformers torch
"""

import argparse
import csv
import json
import os
import sys
import logging
from itertools import combinations
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def discover_problems(work_dir: str, sample_nums: List[int]) -> Set[str]:
    """
    Discover problem directory names present across samples.

    Returns:
        Set of problem directory names (e.g. "cvdp_copilot_64b66b_decoder")
    """
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
    """
    Load all generated code files for a problem+sample by walking the harness directory.

    Concatenates all .sv / .v / .svh / .vh files found under harness/ into a single string,
    sorted by path for deterministic ordering.

    Returns:
        Concatenated code string, or empty string if nothing found.
    """
    harness_dir = os.path.join(work_dir, f"sample_{sample}", problem_dir, "harness")
    if not os.path.isdir(harness_dir):
        return ""

    code_extensions = {".sv", ".v", ".svh", ".vh", ".vhd", ".vhdl"}
    # Exclude generated simulation artifacts
    exclude_patterns = {"cocotb_iverilog_dump.v", "sim_build", "rundir"}

    code_files = []
    for root, dirs, files in os.walk(harness_dir):
        # Skip simulation build directories
        dirs[:] = [d for d in dirs if d not in exclude_patterns]

        # Only look in rtl/ subdirectories (not src/, verif/, rundir/)
        rel_root = os.path.relpath(root, harness_dir)
        path_parts = rel_root.split(os.sep)
        # Valid paths: "{issue}/rtl/..." or "{issue}/rtl"
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

    # Sort by relative path for deterministic concatenation
    code_files.sort(key=lambda x: x[0])
    return "\n".join(content for _, content in code_files)


def compute_scores(
    candidates: List[str],
    references: List[str],
    device: str = "cpu",
    batch_size: int = 32,
) -> Tuple[List[float], List[float]]:
    """
    Compute BERTScore and CodeBERTScore F1 for candidate-reference pairs.

    Returns:
        Tuple of (bert_f1_list, codebert_f1_list)
    """
    from bert_score import score as bert_score_fn

    logger.info("Computing BERTScore (roberta-large) for %d pairs...", len(candidates))
    _, _, F1_bert = bert_score_fn(
        candidates,
        references,
        lang="en",
        device=device,
        batch_size=batch_size,
        verbose=False,
    )

    logger.info("Computing CodeBERTScore (microsoft/codebert-base) for %d pairs...", len(candidates))
    _, _, F1_code = bert_score_fn(
        candidates,
        references,
        model_type="microsoft/codebert-base",
        num_layers=12,
        device=device,
        batch_size=batch_size,
        verbose=False,
    )

    return F1_bert.tolist(), F1_code.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Compute inter-sample pair-wise BERTScore and CodeBERTScore.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_code_similarity.py --work-dir work
  python compute_code_similarity.py --work-dir work --device cuda
  python compute_code_similarity.py --work-dir work -o results/my_scores
        """,
    )
    parser.add_argument(
        "--work-dir", type=str, default="work",
        help="Root work directory containing sample_N subdirectories (default: work)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file prefix (default: <work-dir>/inter_sample_similarity). "
             "Produces .csv and .json files.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for model inference (default: cpu)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for BERTScore computation (default: 32)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        print(f"Error: Work directory not found: {args.work_dir}")
        sys.exit(1)

    output_prefix = args.output or os.path.join(args.work_dir, "inter_sample_similarity")
    output_csv = output_prefix + ".csv"
    output_json = output_prefix + ".json"

    # Discover available samples
    sample_nums = sorted(
        int(d.split("_")[1])
        for d in os.listdir(args.work_dir)
        if d.startswith("sample_") and os.path.isdir(os.path.join(args.work_dir, d))
    )
    if len(sample_nums) < 2:
        print(f"Error: Need at least 2 samples for inter-sample comparison, found {len(sample_nums)}")
        sys.exit(1)

    sample_pairs = list(combinations(sample_nums, 2))
    logger.info("Samples: %s  ->  %d pairs: %s", sample_nums, len(sample_pairs), sample_pairs)

    # Discover all problems across samples
    problem_dirs = sorted(discover_problems(args.work_dir, sample_nums))
    logger.info("Discovered %d problem directories", len(problem_dirs))

    # ----------------------------------------------------------------
    # Phase 1: Load code and build scoring batch
    # ----------------------------------------------------------------
    batch_candidates: List[str] = []
    batch_references: List[str] = []
    scoring_map: List[Tuple[str, Tuple[int, int]]] = []  # (problem_dir, (si, sj))

    problems_with_code = 0
    problems_skipped = 0

    for problem_dir in problem_dirs:
        # Load code for each sample
        sample_code: Dict[int, str] = {}
        for s in sample_nums:
            code = load_generated_code(args.work_dir, problem_dir, s)
            if code.strip():
                sample_code[s] = code

        available = [s for s in sample_nums if s in sample_code]
        if len(available) < 2:
            problems_skipped += 1
            continue

        problems_with_code += 1

        for si, sj in sample_pairs:
            if si in sample_code and sj in sample_code:
                batch_candidates.append(sample_code[si])
                batch_references.append(sample_code[sj])
                scoring_map.append((problem_dir, (si, sj)))

    logger.info(
        "Problems with code in >=2 samples: %d, skipped: %d",
        problems_with_code, problems_skipped,
    )

    if not batch_candidates:
        print("Error: No inter-sample pairs found to compare.")
        sys.exit(1)

    logger.info("Total pairs to score: %d", len(batch_candidates))

    # ----------------------------------------------------------------
    # Phase 2: Compute scores
    # ----------------------------------------------------------------
    bert_f1_list, codebert_f1_list = compute_scores(
        batch_candidates, batch_references,
        device=args.device,
        batch_size=args.batch_size,
    )

    # ----------------------------------------------------------------
    # Phase 3: Organize results
    # ----------------------------------------------------------------
    results: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {}
    for idx, (problem_dir, pair) in enumerate(scoring_map):
        if problem_dir not in results:
            results[problem_dir] = {}
        results[problem_dir][pair] = {
            "bert_f1": round(bert_f1_list[idx], 6),
            "codebert_f1": round(codebert_f1_list[idx], 6),
        }

    # ----------------------------------------------------------------
    # Phase 4: Write CSV — rows=problems, columns=sample pairs
    # ----------------------------------------------------------------
    pair_labels = [f"({si},{sj})" for si, sj in sample_pairs]

    header = ["problem_id"]
    for lbl in pair_labels:
        header.append(f"bert_f1_{lbl}")
    for lbl in pair_labels:
        header.append(f"codebert_f1_{lbl}")

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for problem_dir in sorted(results.keys()):
            row = [problem_dir]
            for pair in sample_pairs:
                if pair in results[problem_dir]:
                    row.append(results[problem_dir][pair]["bert_f1"])
                else:
                    row.append("")
            for pair in sample_pairs:
                if pair in results[problem_dir]:
                    row.append(results[problem_dir][pair]["codebert_f1"])
                else:
                    row.append("")
            writer.writerow(row)

    logger.info("Saved CSV to %s", output_csv)

    # ----------------------------------------------------------------
    # Phase 5: Write JSON
    # ----------------------------------------------------------------
    def _mean(vals):
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    all_bert = [s["bert_f1"] for pid_scores in results.values() for s in pid_scores.values()]
    all_codebert = [s["codebert_f1"] for pid_scores in results.values() for s in pid_scores.values()]

    per_problem_summary = {}
    for pid in sorted(results.keys()):
        bert_vals = [s["bert_f1"] for s in results[pid].values()]
        codebert_vals = [s["codebert_f1"] for s in results[pid].values()]
        per_problem_summary[pid] = {
            "n_pairs": len(bert_vals),
            "bert_f1_mean": _mean(bert_vals),
            "bert_f1_min": round(min(bert_vals), 6) if bert_vals else None,
            "bert_f1_max": round(max(bert_vals), 6) if bert_vals else None,
            "codebert_f1_mean": _mean(codebert_vals),
            "codebert_f1_min": round(min(codebert_vals), 6) if codebert_vals else None,
            "codebert_f1_max": round(max(codebert_vals), 6) if codebert_vals else None,
        }

    json_output = {
        "summary": {
            "num_problems": len(results),
            "num_samples": len(sample_nums),
            "sample_pairs": [list(p) for p in sample_pairs],
            "total_scored_pairs": len(all_bert),
            "bert_f1": {
                "mean": _mean(all_bert),
                "min": round(min(all_bert), 6) if all_bert else None,
                "max": round(max(all_bert), 6) if all_bert else None,
            },
            "codebert_f1": {
                "mean": _mean(all_codebert),
                "min": round(min(all_codebert), 6) if all_codebert else None,
                "max": round(max(all_codebert), 6) if all_codebert else None,
            },
        },
        "per_problem": per_problem_summary,
        "per_pair": {
            pid: {
                f"({si},{sj})": scores
                for (si, sj), scores in pid_scores.items()
            }
            for pid, pid_scores in sorted(results.items())
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    logger.info("Saved JSON to %s", output_json)

    # ----------------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("INTER-SAMPLE CODE SIMILARITY (BERTScore & CodeBERTScore)")
    print("=" * 70)
    print(f"Work dir:      {args.work_dir}")
    print(f"Samples:       {sample_nums}")
    print(f"Sample pairs:  {len(sample_pairs)}")
    print(f"Problems:      {len(results)}")
    print(f"Scored pairs:  {len(all_bert)}")
    print("-" * 70)
    print(f"{'Metric':<25} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    s = json_output["summary"]
    print(f"{'BERTScore F1':<25} {s['bert_f1']['mean']:>10.4f} "
          f"{s['bert_f1']['min']:>10.4f} {s['bert_f1']['max']:>10.4f}")
    print(f"{'CodeBERTScore F1':<25} {s['codebert_f1']['mean']:>10.4f} "
          f"{s['codebert_f1']['min']:>10.4f} {s['codebert_f1']['max']:>10.4f}")
    print("=" * 70)
    print(f"\nResults saved to:\n  CSV:  {output_csv}\n  JSON: {output_json}")


if __name__ == "__main__":
    main()
