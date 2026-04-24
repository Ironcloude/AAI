from datetime import datetime
import json
import sys
from pathlib import Path
import papermill as pm
sys.path.append(str(Path(__file__).resolve().parent.parent))
from experiment_configs import task_1_config as experiments

NOTEBOOK_IN = Path(__file__).parent / "task_1_train.ipynb"
OUTPUT_DIR = Path(__file__).parent / "runs" / "papermill"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS={
    "instacart_subset": "data/insta_clean_data.csv",
}

SEEDS=[43]

EXPERIMENTS=[
    experiments.ncf_baseline,
    experiments.ncf_deep,
    experiments.lstm_baseline,
    experiments.lstm_long_context,
    experiments.sasrec_baseline,
    experiments.sasrec_heavy
]

def ensure_parameters_tag(notebook_path: Path) -> None:
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found at {notebook_path}")

    notebook = json.loads(notebook_path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])

        if 'RUN_ALL=""' in src or 'RUN_ALL=""' in src:
            tags = cell.setdefault("metadata",{}).setdefault("tags",[])
            if "parameters" not in tags:
                tags.append("parameters")
                notebook_path.write_text(json.dumps(notebook,indent=1)+"\n")
                print(f"Re-tagged parameters cell in {notebook_path.name}")
            return
    raise Exception(f"No RUN_ALL parameters cell found in {notebook_path}")

if __name__ == "__main__":
    ensure_parameters_tag(NOTEBOOK_IN)
    runs=[
        (experiment,label,path,seed)
        for experiment in EXPERIMENTS
        for label,path in DATASETS.items()
        for seed in SEEDS
    ]

    if len(runs) > 0:
        print("\nQueueing the following runs:")
        for r in runs:
            print(f" - {r[0].display_name} on {r[1]} (Seed: {r[3]})")
        confirm = input(f"\nAbout to run {len(runs)} experiments. Press Enter to confirm...")
    
    total=len(runs)
    for i, (experiment, dataset_label, dataset_path, seed) in enumerate(runs, start=1):
        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = OUTPUT_DIR / f"{experiment.display_name}_{dataset_label}_{timestamp}.ipynb"
        print(f"\n=== [{i}/{total}] Running {experiment.display_name} ===")

        try:
            pm.execute_notebook(
                str(NOTEBOOK_IN),
                str(out),
                parameters={
                    "RUN_ALL": experiment.display_name,
                    "DATASET_PATH": dataset_path,
                    "VARIABLE_SEED": seed
                },
                kernel_name="python3",
                cwd=str(NOTEBOOK_IN.parent),
                progress_bar=True
            )
            print(f"Done! Output saved to {out.name}")
        except Exception as e:
            print(f"!!! {experiment.display_name} failed: {e}")
            continue
