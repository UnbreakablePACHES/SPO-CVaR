import argparse
import subprocess
import sys
from pathlib import Path


CONFIG_ROOT = Path("configs/pre_exp")
OUTPUT_ROOT = Path("outputs/pre_exp")
DATASET_CONFIGS = {
    "ETF_A": CONFIG_ROOT / "pre_exp_ETF_A.yaml",
    "ETF_B": CONFIG_ROOT / "pre_exp_ETF_B.yaml",
    "DOW": CONFIG_ROOT / "pre_exp_DOW.yaml",
}
DATASET_ORDER = ["ETF_A", "ETF_B", "DOW"]
LAMBDA_RISKS = [0.1, 1, 10, 20, 50]
DEFAULT_SEEDS = list(range(42, 47))


def collect_tasks(datasets=None, lambda_risks=None, seeds=None):
    datasets = datasets or DATASET_ORDER
    lambda_risks = lambda_risks or LAMBDA_RISKS
    seeds = seeds or DEFAULT_SEEDS

    tasks = []
    for dataset in datasets:
        config_path = DATASET_CONFIGS[dataset]
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")
        for lambda_risk in lambda_risks:
            for seed in seeds:
                tasks.append((dataset, config_path, float(lambda_risk), int(seed)))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run pre-exp SPO Markowitz risk-lambda sweep without baselines. "
            "Defaults: lambda_risk in {0.1,1,10,20,50}, seeds 42-46."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch run.py. Defaults to current interpreter.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_ORDER,
        help="Run only this dataset. Can be passed multiple times.",
    )
    parser.add_argument(
        "--lambda-risk",
        dest="lambda_risks",
        action="append",
        type=float,
        help="Run only this lambda_risk. Can be passed multiple times.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Run only this seed. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Root output directory. Results are written to output-root/dataset/seed_<seed>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining tasks if one task fails.",
    )
    args = parser.parse_args()

    tasks = collect_tasks(args.dataset, args.lambda_risks, args.seed)
    output_root = Path(args.output_root)

    for i, (dataset, config_path, lambda_risk, seed) in enumerate(tasks, start=1):
        output_dir = output_root / dataset / f"seed_{seed}"
        cmd = [
            args.python,
            "run.py",
            "--config",
            str(config_path),
            "--model_type",
            "markowitz",
            "--lambda_risk",
            str(lambda_risk),
            "--seed",
            str(seed),
            "--output_dir",
            str(output_dir),
            "--skip_baselines",
        ]

        print(
            f"\n[{i}/{len(tasks)}] dataset={dataset} "
            f"lambda_risk={lambda_risk:g} seed={seed}"
        )
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            message = (
                f"failed: dataset={dataset} lambda_risk={lambda_risk:g} "
                f"seed={seed} config={config_path}"
            )
            if args.continue_on_error:
                print(message, file=sys.stderr)
            else:
                raise SystemExit(message)


if __name__ == "__main__":
    main()
