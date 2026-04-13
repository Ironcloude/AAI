# AI marketplace scaffold

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

### Health

```bash
curl http://localhost:8000/health
curl http://localhost:5000/health
```

### List models

```bash
curl http://localhost:8000/models
curl http://localhost:5000/models
```

### Tabular prediction

```bash
curl -X POST http://localhost:5000/predict/tabular \
  -H 'Content-Type: application/json' \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### Image prediction

```bash
curl -X POST http://localhost:5000/predict/image \
  -F 'image=@sample.jpg'
```

### Upload a replacement tabular model

The AI service can execute uploaded `.joblib` or `.pkl` sklearn-compatible tabular models. It will also accept `.safetensors` and `.gguf` files into the registry, but they are metadata-only until you add a loader/runtime for those model types.

```bash
curl -X POST http://localhost:5000/models/upload \
  -F 'model=@my_model.joblib' \
  -F 'task_type=tabular' \
  -F 'display_name=my-model'
```

### Select active model

```bash
curl -X POST http://localhost:5000/models/select \
  -H 'Content-Type: application/json' \
  -d '{"model_name": "my_model.joblib"}'
```

## Notes

- No database integration is included.
- The image endpoint uses a heuristic rubric so the service is runnable now.
- The tiny startup model is trained from the Iris dataset and written to `/app/models/tiny-iris-logreg.joblib` on first boot.
