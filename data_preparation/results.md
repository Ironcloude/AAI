# AI Grocery Demand Forecasting & Sequence Recommendation

## 1. Project Overview
This project solves two critical but distinct problems in the grocery retail domain: **Inventory Management** (Forecasting) and **User Personalization** (Recommendation). 

Initially, the architecture conflated sequential behavioral data with chronological time-series data, resulting in flawed, synthetic cycles. The system has been completely re-architected to split these tasks using the appropriate datasets, models, and evaluation metrics.

1. **Demand Forecasting (The "When" & "How Much"):** Predicts aggregate daily volume for specific products to drive supply chain and warehouse decisions.
2. **Product Recommendation (The "What" & "Who"):** Learns from individual user purchase histories to predict the exact next item they will add to their cart.

---

## 2. Model Parameter Justifications (`config.py`)

To ensure scientific cross-validation, the hyperparameter grids are explicitly defined into three distinct series (`A`, `B`, and `C`). The parameters were chosen specifically to handle the mathematical nature of grocery data.

### A-Series: Forecasting LSTM (`ForecastingLSTMConfig`)
*Applies Deep Learning to chronological daily demand (groceries_dataset.csv).*

*   **`lookback_window` (14 & 21 days):** Grocery data is heavily dictated by 7-day weekly cycles (e.g., Saturdays behave similarly to previous Saturdays). The lookback is strictly locked to multiples of 7 (2 or 3 weeks) to ensure the neural network receives full cyclic context.
*   **`hidden_size` (64 & 128) & `num_layers` (1 & 2):** Time-series data is fundamentally 1-Dimensional. Using massive networks (e.g., 512+ units) causes catastrophic overfitting on 1D data. Keeping the capacity tight forces the LSTM to learn generalized trends rather than memorizing noise.
*   **`future_steps` (14 days):** Represents a realistic 2-week operational horizon for retail supply chain ordering.

### B-Series: Forecasting SARIMAX (`ForecastingSARIMAXConfig`)
*Applies Statistical ARIMA modeling to the exact same chronological daily dataset.*

*   **`seasonal_order` `(P, D, Q, s=7)`:** The fundamental parameter here is `s=7`. Because the time index was forced into a strict daily frequency (filling missing days with 0), we can explicitly tell the solver that seasonality occurs exactly every 7 days.
*   **`order` `(p, d, q)`:** Initialized at standard bounds `(1,0,1)` and `(2,1,2)` to account for immediate day-to-day momentum (Auto-Regressive) and sudden demand shocks (Moving Average). 

### C-Series: Recommender LSTM (`RecommenderLSTMConfig`)
*Treats user purchase history as categorical NLP-style sequences (insta_clean_data.csv).*

*   **`max_lookback` (20 items):** When predicting the next item a user will buy, ancient history is less relevant than recent context. A lookback of 20 items captures roughly the user's last 1 to 2 complete shopping baskets.
*   **`embed_dim` (64 & 128):** Converts discrete, categorical product IDs into dense vector representations. This allows the model to learn that "Oat Milk" and "Almond Milk" exist close to each other in latent space.
*   **`hidden_dim` (128 & 256):** Noticeably larger than the Forecasting LSTM. This is because the Recommender must learn complex, multidimensional relationships across thousands of unique products (e.g., Bread $\rightarrow$ Peanut Butter $\rightarrow$ Jelly), requiring more network capacity than a 1D volume forecaster.

---

## 3. Engineering & Security Architecture

### A. Security First (`safetensors`)
Standard PyTorch `.pth` files use Python `pickle`, which allows arbitrary code execution. Because these models are designed for production, all Deep Learning weights are exported using HuggingFace's `safetensors`. *(Note: SARIMAX relies on `.pkl` out of necessity, as it is a complex Statsmodels Python wrapper, not a raw tensor array).*

### B. Self-Contained Output Dashboards
To prevent directory clutter (e.g., hundreds of `_results.md` files), the reporting system (`run_utils.py`) dynamically compiles results into a **single interactive HTML file** per run:
1.  **Plotly Graphs:** Interactive visualizations of the 14-day future inference or epoch loss curves.
2.  **Embedded Markdown:** A styled `<pre>` code block containing the exact Hyperparameter Config and Evaluation Metrics (MAE, RMSE, Accuracy).
3.  **Clipboard Integration:** A JavaScript-powered "Copy Markdown" button allows engineers to instantly pull the formatted data tables into their own documentation or LLM agents without needing secondary files.

### C. True Future Inference
The time-series forecasting scripts do not simply stop at evaluating the test set. They execute an algorithmic `torch.no_grad()` sliding-window loop that feeds the model's own predictions back into itself, effectively stepping into the unknown future to forecast the required `14` days ahead.