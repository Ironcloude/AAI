"""
Batch-run task_2_train.ipynb across (experiment x dataset) pairs via papermill.
Usage (wsl):  cd task_2 ; CUDA_LAUNCH_BLOCKING=1 uv run python run_all.py 
Usage (win): cd task_2 ; uv run python run_all.py
"""
from datetime import datetime
import json
import sys
from pathlib import Path
import papermill as pm
sys.path.append(str(Path(__file__).resolve().parent.parent))
from experiment_configs import task_2_config_final as experiments

NOTEBOOK_IN = Path(__file__).parent / "task_2_train.ipynb"
OUTPUT_DIR = Path(__file__).parent / "runs" / "papermill"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Label -> dataset path passed to the notebook as DATASET_PATH.
# Label is appended to the output notebook name so runs don't collide.
DATASETS = {
    "clean" : "Fruit_And_Vegetable_Diseases_Dataset_no_identical_no_aug", # deduplicated, no augmentation
}

EXPERIMENTS = [
    experiments.EX1_EFFICIENTNET_FINETUNE,
    experiments.EX2_SWIN_FINETUNE,
    experiments.EX3_MAXVIT_FINETUNE,
    experiments.EX1T_EFFICIENTNET_FINETUNE,
    experiments.EX2T_SWIN_FINETUNE,
]

def ensure_parameters_tag(notebook_path: Path) -> None:
    """Guarantee the RUN_ALL cell is tagged 'parameters' for papermill.

    VSCode/Jupyter UI saves occasionally strip custom tags. Re-apply on each
    launch so the notebook is always in the expected state.
    """
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if 'RUN_ALL = ""' in src:
            tags = cell.setdefault("metadata", {}).setdefault("tags", [])
            if "parameters" not in tags:
                tags.append("parameters")
                notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
                print(f"Re-tagged parameters cell in {notebook_path.name}")
            return
    raise Exception(f"No RUN_ALL parameters cell found in {notebook_path}")

if __name__ == "__main__":
    ensure_parameters_tag(NOTEBOOK_IN)

    runs = [(experiment, label, path) for experiment in EXPERIMENTS for label, path in DATASETS.items()]
    total = len(runs)
    for i, (experiment, dataset_label, dataset_path) in enumerate(runs, start=1):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = OUTPUT_DIR / f"{experiment.display_name}_{dataset_label}_{timestamp}.ipynb"
        print(f"\n=== [{i}/{total}] {experiment.display_name} | dataset={dataset_label} -> {out} ===")
        try:
            pm.execute_notebook(
                str(NOTEBOOK_IN),
                str(out),
                parameters={"RUN_ALL": experiment.display_name, "DATASET_PATH": dataset_path},
                kernel_name="python3",
                cwd=str(NOTEBOOK_IN.parent),
                progress_bar=True,
            )
        except Exception as e:
            print(f"!!! {experiment.display_name} | {dataset_label} failed: {e}")
            continue
