"""
Comparison and Benchmarking Tool.
Aggregates metrics from multiple experiments and generates comparative markdown tables.
"""

import argparse
import glob
import json
import os
from typing import Dict, List


def load_metrics_for_experiments(artifacts_dir: str = "artifacts") -> Dict[str, dict]:
    """Load all metrics.json files from subdirectories in artifacts/."""
    experiments = {}
    pattern = os.path.join(artifacts_dir, "*", "metrics.json")
    for metrics_file in glob.glob(pattern):
        exp_name = os.path.basename(os.path.dirname(metrics_file))
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            experiments[exp_name] = data
        except Exception as e:
            print(f"Failed to read {metrics_file}: {e}")
    return experiments


def generate_comparison_table(experiments: Dict[str, dict]) -> str:
    """Generate GitHub Flavored Markdown comparison tables."""
    if not experiments:
        return "No experiment metrics found."

    lines = []
    lines.append("### 🏆 Architecture Comparison Summary\n")
    lines.append(
        "| Architecture | Test Accuracy | Macro F1 | Weighted F1 | Glioma Recall | Latency (ms) | Throughput (FPS) | Params (M) |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for name, m in experiments.items():
        acc = f"{m.get('test_accuracy', 0.0) * 100:.2f}%"
        macro_f1 = f"{m.get('macro_f1', 0.0):.4f}"
        weighted_f1 = f"{m.get('weighted_f1', 0.0):.4f}"
        glioma_rec = f"{m.get('per_class', {}).get('glioma', {}).get('recall', 0.0):.4f}"
        latency = f"{m.get('inference_benchmark', {}).get('mean_latency_ms', 0.0):.1f}"
        fps = f"{m.get('inference_benchmark', {}).get('throughput_fps', 0.0):.1f}"
        params = f"{m.get('parameters', {}).get('total', 0) / 1e6:.2f}M"

        lines.append(
            f"| **{name}** | {acc} | {macro_f1} | {weighted_f1} | {glioma_rec} | {latency} | {fps} | {params} |"
        )

    lines.append("\n### 🔬 Per-Class F1-Score Breakdown\n")
    lines.append("| Architecture | Glioma F1 | Meningioma F1 | No Tumor F1 | Pituitary F1 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for name, m in experiments.items():
        pc = m.get("per_class", {})
        g_f1 = f"{pc.get('glioma', {}).get('f1_score', 0.0):.4f}"
        m_f1 = f"{pc.get('meningioma', {}).get('f1_score', 0.0):.4f}"
        n_f1 = f"{pc.get('notumor', {}).get('f1_score', 0.0):.4f}"
        p_f1 = f"{pc.get('pituitary', {}).get('f1_score', 0.0):.4f}"
        lines.append(f"| **{name}** | {g_f1} | {m_f1} | {n_f1} | {p_f1} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare NeuroMap model metrics.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Artifacts root directory")
    parser.add_argument("--save-md", type=str, default=None, help="Save report to markdown file")
    args = parser.parse_args()

    exps = load_metrics_for_experiments(args.artifacts_dir)
    report = generate_comparison_table(exps)
    print(report)

    if args.save_md:
        with open(args.save_md, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport written to {args.save_md}")


if __name__ == "__main__":
    main()
