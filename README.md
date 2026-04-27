# Advanced Artificial Intelligence
Repository for the AAI group project.

Notebooks:

Task 1
- [Task1 part_1](./task_1/dataset_groceries.ipynb)
- [Task1 part_2](./task_1/dataset_instacart.ipynb)
- [Task1 part_3](./task_1/task1part3.ipynb)

Task 2

- [data_eval.ipynb](./task_2/data_eval.ipynb)
- [golden_dataset_eval.ipynb](./task_2/golden_dataset_eval.ipynb)
- [task_2_inference.ipynb](./task_2/task_2_inference.ipynb)
- [task_2_optuma.ipynb](./task_2/task_2_optuma.ipynb)
- [task_2_train.ipynb](./task_2/task_2_train.ipynb)

<b>Models, datasets and other ML related resources can be found [here](https://drive.google.com/drive/folders/1kz25InaRNNnn0-tkePQ_GyJM4yTLSm8O?usp=sharing).</b>

```bash
git clone <this_repo>
cd <this_repo>
uv venv --python 3.11

# Activate uv
## MacOS & Linux
source .venv/bin/activate

## Windows
.\.venv\Scripts\activate

# Install base requirements 
uv pip install -r requirements.txt
# [Optional] For GPU acceleration (find specific version here: https://pytorch.org/get-started/locally/)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Training metrics log to Weights & Biases (one-time):
wandb login # https://wandb.ai/jaimespencer2-/projects
```



### Project Structure & Responsibility
```text
task_2/
├── data/                       # Datasets, masks, and serialized reference files
│   ├── colour_references_rembg.pkl  # Pickled HSV histograms for healthy fruit types
│   ├── generic_colour_distribution.pkl # Empirical CDFs for vibrancy, brightness, and uniformity
│   ├── README.md               # Documentation for the data directory
│   ├── rembg_masks/            # Segmentation masks generated via rembg
│   └── test_data/              # Sample images for testing and inference
├── figures/                    # Analysis plots and result visualizations
│   ├── colour_references_per_type.png # Visual summary of fruit-specific colour baselines
│   ├── generic_colour_ecdfs.html # Interactive Plotly eCDF curves for colour components
│   └── generic_colour_ecdfs.png  # Static snapshot of the eCDF curves
├── __init__.py                 # Makes task_2 a Python package
├── models/                     # Saved model checkpoints
│   └── EX3_efficientnet_finetune_mtl_20260414183210.safetensors # Pre-trained MTL weights
├── runs/                       # Experiment logs and metadata
│   └── README.md               # Documentation for tracking experiments
├── task_2_inference.ipynb      # Notebook for quality assessment and model inference
├── task_2_train.ipynb          # Notebook for training the Multi-Task Learning model
└── utils/                      # Core logic and helper modules
    ├── build_references.py     # Script to calculate baseline colour stats from healthy produce
    ├── data_count.py           # Utility to print dataset class distribution
    ├── dataset.py              # PyTorch Dataset for produce health/type classification
    ├── generate_masks.py       # Segmentation logic using SAM2, rembg, or GrabCut
    ├── grade_produce.py        # Logic for colour and shape (solidity) quality grading
    ├── __init__.py             # Makes utils a Python package
    ├── mtl_model.py            # MultiTaskClassifier architecture (backbone + dual heads)
    └── utils.py                # Generic helpers like reproducibility seeding
```



## TASK 2: Image Classification
|   |   |   | 
|---|---|---|
|  **Dataset**  | [Fruit and Vegetable Disease (Healthy vs Rotten)](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten)  | -- | 
|  **Library** | Pytorch (Stable 2.10) | [Download](https://pytorch.org/get-started/locally/) |
| **Model list** | [Pytorch models and leaderboard](https://docs.pytorch.org/vision/stable/models.html) | -- |

<details>
<summary>
  
## Models
  
</summary>

|  Model | Architecture  |  Notes |   
|---|---|---|
|  [**EfficientNetV2**](https://arxiv.org/pdf/2104.00298)|  CNN  | <ul><li>Top-performing models at a fraction of performance.<li>Data efficient</li><li>Better heatmaps than transformers (XAI)</li></ul> |
|  Swin | Transformer  | <ul><li>Data hungry but possibly better performance.</li><li>Allegedly poorer XAI</li></ul>|
| MaxVit_T | Hybrid | -- |

## Key variants

Models variants across the three architectures with comporable _GFLOPS_ were selected.
- There are several pytorch models & variants with slightly greater performance but computation is many magnitudes higher.
  
| Model  | Architecture | ACC | PARAMS | GFLOPS |  Purpose | Notes |
|---|---|---|---|---|---|---|
| EfficientNet_V2_S_Weights.IMAGENET1K_V1 |  CNN   | 	84.228 | 	21.5M | 	 **8.37**| Arch comparison | -- |
| EfficientNet_v2_S (No pre-training) | CNN |  -- | 21.5M ~ | 21.5M ~ | No transfer-learning  | -- | 
| Swin_S_Weights.IMAGENET1K_V1 | Transformer  | 	83.196 | 49.6M | **8.74**| Arch comparison | -- |
| MaxVit_T  | Hybrid | 83.7 | 30.9M | **5.56**| Arch comparison | Only variant | 
| EfficientNet_V2_L_Weights.IMAGENET1K_V1 | CNN |  85.808 | 118.5M |**56.08** |Performance | -- |

</details>
<details>
<summary>

## Key hyperparameters and variables

</summary>

| Parameter | Example | Arch |
|---|---|---|
|Loss function| CrossEntropy, WeightedCrossEntropy | --- |
| Optimizer | AdamW, SGD, RMSprop | --- |
| Learning rate (LR) | --- | --- |
| Learning rate schedular | --- | --- |
| Epochs | --- | --- |
| Batch size | --- |  --- |
| Weight decay | --- | --- |
| Kernel Size | --- | CNN |
| Pooling type | --- | CNN |
| Stride | --- | CNN |
| Padding | --- | CNN |
| Filter count | --- | CNN |
| Hidden size | --- | Transformer |
| Attention heads | --- | Transformer |
| Transformer layers | --- | Transformer |
| Patch size | --- | Transformer |
- Transfer learning vs. fine-tuning
- Additional can experiment with partial freezing for transfer learning.

</details>
<details>
<summary>
  
## XAI

</summary>

| Tool | Use |
|---|---|
| Grad-cam |  Visual heatmap of crucial image regions; what the "model is looking at". Made for CNNs. Optimal for image classification. |
| LIME | Local feature importance. Explanations may slightly differ for the same prediction. Lightweight | 
| SHAP |  Local (one prediction) or global (model-wide) feature importance. Consistent. Expensive for images. |
<img width="567" height="440" alt="image" src="https://github.com/user-attachments/assets/84639a7d-d421-4fa8-ba9b-054f36d16f53" />

</details>
<details>
<summary>
  
## Evaluation

</summary>

### Quantitative

| Metric | Purpose |
|---|---|
| Accuracy | --- |
| Validation Loss | --- |
| Recall | Evaluate false negatives |
| F1 | Balance of precision and recall |
| Confusion Matrix | --- |
| AUC-ROC (Binary) | Measure of distinguishing difference | 

### Qualitative
- Effects of finetuning (with, without and different methods)
- Effect of pretraining (with and without)
- Comparison of model architectures

</details>
