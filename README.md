# Advanced Artificial Intelligence
Repository for AAI group project.
|   |   |   | 
|---|---|---|
|  **Dataset**  | [Fruit and Vegetable Disease (Healthy vs Rotten)](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten)  | -- | 
|  **Library** | Pytorch (Stable 2.10) | [Download](https://pytorch.org/get-started/locally/) |
| **Model list** | [Pytorch models and leaderboard](https://docs.pytorch.org/vision/stable/models.html) | -- |

# Models

|  Model | Architecture  |  Notes |   
|---|---|---|
|  [**EfficientNetV2**](https://arxiv.org/pdf/2104.00298)|  CNN  | <ul><li>Top-performing models at a fraction of performance.<li>Data effecient</li><li>Better heatmaps than transformers (XAI)</li></ul> |
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


## Key hyperparamters
| Parameter | Example | Arch |
|---|---|---|
|Loss function| CrossEntropy, WeightedCrossEntropy | --- |
| Optimizer | AdamW, SGD, RMSprop | --- |
| Learning rate (LR) | --- | --- |
| Batch size | --- |  --- |
| Weight decay | --- | --- |
| Kernel Size | --- | CNN |
| Stride | --- | CNN |
| Padding | --- | CNN |
| Filter count | --- | CNN |
| Hidden size | --- | Transformer |
| Attention heads | --- | Transformer |
| Transformer layers | --- | Transformer |

## XAI

## Evaluation
