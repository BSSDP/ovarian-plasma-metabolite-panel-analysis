from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("OV_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
ROOT = PROJECT_ROOT / "12_scFEA"
SCFEA = ROOT / "01_software" / "scFEA"
RUNNER = SCFEA / "src" / "scFEA_project.py"
DATA_DIR = SCFEA / "data"
DATASETS = ["GSE217517", "GSE184880"]


def run_one(dataset: str, mode: str, epochs: int, impute: bool) -> None:
    input_dir = ROOT / "02_inputs" / f"{dataset}_epithelial"
    result_dir = ROOT / "03_results" / f"{dataset}_epithelial" / mode
    result_dir.mkdir(parents=True, exist_ok=True)
    result_dir_for_scfea = os.path.relpath(result_dir, SCFEA)
    stem = f"{dataset}_epithelial" if mode == "full" else f"{dataset}_epithelial_pilot500"
    input_file = input_dir / f"{stem}_scFEA_counts.csv"
    flux_file = result_dir / f"{stem}_flux_m168.csv"
    balance_file = result_dir / f"{stem}_metabolite_balance_c70.csv"
    log_file = result_dir / f"{stem}_run.log"
    command = [
        sys.executable,
        str(RUNNER),
        "--data_dir",
        str(DATA_DIR),
        "--input_dir",
        str(input_dir),
        "--res_dir",
        result_dir_for_scfea,
        "--test_file",
        input_file.name,
        "--moduleGene_file",
        "module_gene_m168.csv",
        "--stoichiometry_matrix",
        "cmMat_c70_m168.csv",
        "--cName_file",
        "cName_c70_m168.csv",
        "--sc_imputation",
        str(impute),
        "--output_flux_file",
        str(flux_file),
        "--output_balance_file",
        str(balance_file),
        "--train_epoch",
        str(epochs),
    ]
    record = {
        "dataset": dataset,
        "mode": mode,
        "epochs": epochs,
        "sc_imputation": impute,
        "started": datetime.now().isoformat(),
        "command": command,
    }
    env = os.environ.copy()
    env.update(
        {
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    with log_file.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=SCFEA, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    record["returncode"] = completed.returncode
    record["finished"] = datetime.now().isoformat()
    loss_files = sorted((SCFEA / "output").glob("lossValue_*.txt"), key=lambda p: p.stat().st_mtime)
    if loss_files:
        convergence_file = result_dir / f"{stem}_convergence_loss.txt"
        shutil.copy2(loss_files[-1], convergence_file)
        record["convergence_loss_file"] = str(convergence_file)
    (result_dir / f"{stem}_run_manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(f"{dataset} {mode} scFEA failed; see {log_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--no-imputation", action="store_true")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        run_one(dataset, args.mode, args.epochs, not args.no_imputation)


if __name__ == "__main__":
    main()
