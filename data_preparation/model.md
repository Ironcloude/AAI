# Model Runs Summary

This document provides a quick overview of the model runs configured and executed across the forecasting and recommendation pipelines, following a strict naming convention.

### Naming Convention
**Format:** `[PREFIX]` | **DAILY** `[1 | 2]` | **MONTHLY** `[3 | 4]`

**Legend:** 
* `h` = Hidden dimensions
* `L` = Number of Layers
* `LB` = Lookback window
* `S` = Seasonality
* `e` = Embedding dimension

---

### Run Matrix

| Run ID | Model Type | Domain | Frequency | Architecture / Hyperparameters |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | **LSTM** | Forecasting | Daily | 64 Hidden / 1 Layer |
| **A2** | **LSTM** | Forecasting | Daily | 128 Hidden / 2 Layers |
| **A3** | **LSTM** | Forecasting | Monthly | Lookback: 2 |
| **A4** | **LSTM** | Forecasting | Monthly | Lookback: 3 |
| **B1** | **SARIMAX** | Forecasting | Daily | Order (1, 1, 0) |
| **B2** | **SARIMAX** | Forecasting | Daily | Order (2, 1, 2) |
| **B3** | **SARIMAX** | Forecasting | Monthly | No Seasonality |
| **B4** | **SARIMAX** | Forecasting | Monthly | 6-month Seasonality |
| **C1** | **XGBoost** | Forecasting | Daily | Max Depth: 6 |
| **C2** | **XGBoost** | Forecasting | Daily | Max Depth: 8 |
| **C3** | **XGBoost** | Forecasting | Monthly | Max Depth: 4 |
| **C4** | **XGBoost** | Forecasting | Monthly | Max Depth: 6 |
| **E1** | **NCF** | Recommendation | N/A | 64 Embed / 128 Hidden |
| **F1** | **FM** | Recommendation | N/A | Factorization Machine (Base) |
