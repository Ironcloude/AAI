# AI Grocery Demand Forecasting & Sequence Recommendation

## 1. Project Overview & Depth
This project implements a robust machine learning pipeline solving two distinct problems within the grocery retail domain: **Aggregate Demand Forecasting** (Inventory Strategy) and **Personalized Sequence Recommendation** (User Experience). 

The initial architecture suffered from significant algorithmic flaws: it conflated sequential user behavior with chronological time-series data, leading to synthetic 168-hour repeating cycles, and it restricted forecasting to a single item ("whole milk"). 

The system has been completely re-architected into three isolated, highly precise pipelines:
1. **Demand Forecasting (The "When" & "How Much"):** Predicts total aggregate daily/weekly store volume (all items combined) to drive supply chain and warehouse decisions.
2. **Product Recommendation (The "What" & "Who"):** Treats individual user purchase histories as NLP-style sequential tokens to predict the exact next item added to a cart.
3. **Exploratory Data Analysis (EDA):** A completely modular, Jupyter-ready analysis suite to visualize seasonality, autocorrelation (ACF/PACF), and Market Basket combinations.

---

## 2. Architecture & Pipeline Breakdown

### A. Forecasting Pipeline (`forecasting_experiments.py` & `config.py`)
*Dataset: `groceries_dataset.csv`*

The forecasting pipeline now models **Total Daily Grocery Demand**. By switching from a sparse single-product target to total volume, the time series becomes significantly denser and more mathematically learnable.
*   **Chronological Integrity:** The pipeline parses absolute dates (`DD-MM-YYYY`), groups total item volume by date, and critically **reindexes the timeline** using `pd.date_range`. Any calendar days (or weeks) missing from the raw dataset are explicitly filled with `0`, preventing artificial chronological jumps.
*   **True Future Inference:** Both models (LSTM and SARIMAX) do not simply validate on the test set. They execute a recursive loop (e.g., sliding window `torch.no_grad()`) extending 14 days into the literal unknown future.
*   **A-Series Models (LSTM):** PyTorch-based Deep Learning. Utilizes explicit `lookback_window`s of 14 or 21 days to enforce weekly cycle constraints on the neural network.
*   **B-Series Models (SARIMAX):** Statsmodels-based Statistical Modeling. Utilizes `seasonal_order=(P,D,Q,s=7)`, leveraging the strict continuous daily index to guarantee 7-day seasonality is respected.

### B. Recommendation Pipeline (`recommender_lstm.py` & `config.py`)
*Dataset: `insta_clean_data.csv`*

*   **C-Series Models (LSTM):** Treats user history as language. Products are mapped to a `torch.nn.Embedding` layer. A PyTorch LSTM processes the sequences and outputs a `Softmax` distribution across the entire product catalog vocabulary to predict the next sequential item.

### C. Exploratory Data Analysis (`plot_grocery.py` & `flow.txt`)
A modular suite of Plotly-powered functions designed for interactive Jupyter Notebook execution. Key capabilities include:
*   **Time-Series Smoothing:** Aggregates noisy daily volume into weekly chunks (`pd.Grouper(freq='W')`) to reveal true macro-trends.
*   **Autocorrelation (ACF/PACF):** Calculates the mathematical lag-correlation of the time-series, proving the necessity of 7-day seasonality constraints.
*   **Market Basket Analysis:** Implements a strict `get_frequent_pairs` algorithm. It groups baskets by User/Date, strips duplicate items via `set()`, sorts products alphabetically to normalize $(A, B)$ hashing, and calculates combinations via $nCr = \frac{n!}{2!(n-2)!}$.

---

## 3. Engineering & Security Standards

### A. Strict Configuration Management (`config.py`)
All hardcoded arrays and "magic numbers" have been removed. The system utilizes pure Python `@dataclass` structures (`A1`, `A2`, `B1`, `C1`). The models read dynamically from these explicit blueprints, ensuring cross-validation is mathematically verifiable and flawlessly reproducible.

### B. Security First Checkpointing (`safetensors`)
Standard PyTorch `.pth` files utilize Python `pickle`, introducing arbitrary code execution vulnerabilities. Because these models are designed for production deployment, all PyTorch weights (Forecasting LSTM and Recommender LSTM) are exported using HuggingFace's secure `safetensors.torch.save_file`. *(SARIMAX relies on `.pkl` out of absolute necessity, as it is a Statsmodels Python wrapper).*

### C. Self-Contained Output Dashboards (`run_utils.py`)
To prevent directory clutter, the pipeline abandons individual `.md` logging. Instead, it dynamically compiles results into a **single interactive HTML file** per run:
1.  **Interactive Visualizations:** Plotly graphs of the 14-day future inference or epoch loss curves.
2.  **Embedded Markdown Tables:** A styled, dark-mode `<pre>` block containing the exact Hyperparameter Config and Evaluation Metrics (MAE, RMSE, Accuracy).
3.  **Clipboard Integration:** A JavaScript-powered "Copy Markdown" button allows engineers to instantly pull the formatted data tables into their own documentation or LLM context windows with zero friction.