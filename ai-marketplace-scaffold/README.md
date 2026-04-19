# AI marketplace scaffold

<details>
<summary><b> &nbsp Versions </b>
</summary>

2026-04-18:
- Resolved deadlock in model_registry.py caused by concurrent non-reentrant lock acquisition between select_model and load_model, causing activation hangs.
- Fixed bug in app.py where image inference failed to pass the active model to image_pipeline, resulting in fallback to heuristics.
- Updated image_pipeline.py to compute and return ripeness metrics directly from ML model confidence predictions.
</details>


<details>
<summary><b> &nbsp TODO </b>
</summary>
<li>add active model per type (e.g. image classification model should not be used on tabular data)</li>
</details>

### Project Structure & Responsibility

```text
ai-marketplace-scaffold/
├── ai-service/             # Flask inference service & model registry logic
│   ├── app.py              # Defines API endpoints for health, registry, inference
│   ├── bootstrap_model.py  # Initializes the default tabular demo model
│   ├── image_pipeline.py   # Image analysis logic & model-loading hook
│   └── model_registry.py   # Registry management, loading, & XAI implementation
├── web-app/                # Flask proxy UI (Single-page dashboard)
│   ├── app.py              # Relays web requests to ai-service
│   └── templates/index.html# Browser UI for interaction
├── inference.py            # Utility for running model inference outside the service
└── models/                 # Shared storage for model files (.joblib, .pkl, .safetensors, .gguf)
```
*(Note: The `models/` folder contains model artifacts and the `model_registry.json` registry file.)*


This scaffold gives you:
- `ai-service`: Flask inference service with model registry and startup bootstrap for a tiny sklearn model.
- `web-app`: Flask proxy/admin surface with upload and prediction endpoints.
- `models/`: shared bind-mounted model storage.

## Run

```bash
docker compose up --build
```

## Services

- Web app: http://localhost:5000
- AI service: http://localhost:8000

## Quick smoke tests

**Health**
```bash
curl http://localhost:8000/health
curl http://localhost:5000/health
```

**List models**
```bash
curl http://localhost:8000/models
curl http://localhost:5000/models
```

**Tabular prediction**
```bash
curl -X POST http://localhost:5000/predict/tabular \
  -H 'Content-Type: application/json' \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Image prediction**
```bash
curl -X POST http://localhost:5000/predict/image \
  -F 'image=@sample.jpg'
```

**Upload a replacement tabular model**  
The AI service can execute uploaded `.joblib` or `.pkl` sklearn-compatible tabular models. It will also accept `.safetensors` and `.gguf` files into the registry, but they are metadata-only until you add a loader/runtime for those model types.

```bash
curl -X POST http://localhost:5000/models/upload \
  -F 'model=@my_model.joblib' \
  -F 'task_type=tabular' \
  -F 'display_name=my-model'
```

**Select active model**
```bash
curl -X POST http://localhost:5000/models/select \
  -H 'Content-Type: application/json' \
  -d '{"model_name": "my_model.joblib"}'
```

## Notes

- No database integration is included.
- The image endpoint uses a heuristic rubric so the service is runnable now.
- The tiny startup model is trained from the Iris dataset and written to `/app/models/tiny-iris-logreg.joblib` on first boot.

## Concise User Flow

To work with models in this scaffold:

1. **Upload:** Use the Web App (`/models/upload`) to push a model file. It registers the model metadata in `models/model_registry.json`.
2. **Activate:** Use the Web App (`/models/select`) to make the uploaded model the active one.
3. **Infer:** Send data to the Web App endpoints (`/predict/tabular` or `/predict/image`) which proxy to `ai-service` for inference using the active model.

