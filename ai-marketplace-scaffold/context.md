Overview
- Two Flask services on Docker Compose:
  - ai-service (8000): hosts ML endpoints and model registry.
  - web-app (5000): a minimal browser UI and reverse-proxy facade to ai-service.
- Shared models volume: host ./models mounted to /app/models in ai-service.

ai-service
- Purpose: Model registry + inference + simple XAI.
- Key files:
  - ai-service/app.py: registers endpoints.
  - ai-service/model_registry.py: registry state, loading, tabular predict, XAI.
  - ai-service/image_pipeline.py: “heuristic” image analysis + model-loading hook.
- Endpoints:
  - GET /health
  - GET /models
  - POST /models/upload
  - POST /models/select
  - POST /predict/tabular
  - POST /predict/image
  - GET /xai/active-model
- Active model:
  - Only one active model is used for predictions/XAI.
  - POST /models/select sets active_model_name in the registry and preloads if file-backed.
- Model registry (./models/model_registry.json):
  - Minimal schema per model:
    - display_name, task_type (“tabular” | “image”), format (e.g. “.joblib”), path (container path like “/app/models/file”), runtime_supported (bool).
  - Top-level: active_model_name.
- Inference runtime modes (controlled by USE_FILE_MODELS):
  - USE_FILE_MODELS=false (default in compose): Tabular uses DummyTabularModel (in-memory), Image uses heuristic. Fast demos without file I/O.
  - USE_FILE_MODELS=true: Tabular loads .joblib/.pkl via joblib; only models with runtime_supported=true may be activated; Image loader is a stub to be implemented.
- Notable fixes:
  - Robust probabilities: model_registry.predict_tabular now handles both numpy arrays and Python lists (no .tolist() crash).
- Where to implement real image models:
  - image_pipeline.load_model(use_file=True, path=...) should load a real model (e.g., torch/onnx), then pass it into analyse_image.

web-app
- Purpose: Single-page UI + proxy to ai-service (no CORS issues).
- Key files:
  - web-app/app.py: renders index.html and safely relays requests to ai-service (returns JSON or raw text; never crashes on non-JSON).
  - web-app/templates/index.html: simple HTML+fetch UI with minimal JS.
- Environment:
  - AI_SERVICE_URL=http://ai-service:8000 (in docker-compose).
- UI layout (single page, always visible responses):
  - Global status bar under the header showing current action, HTTP code, and reason on failures.
  - Card 1: “Service & Models” (metadata)
    - Sync button (GET /ai/health, GET /models).
    - Displays health dot, active model label.
    - Upload model: file (.joblib, .pkl, .safetensors, .gguf), task type, display name → POST /models/upload.
    - Activate executable model → POST /models/select.
    - Response panels (always open): /ai/health, /models, /models/select.
  - Card 2: “Model operations” (uses model)
    - XAI: GET /xai/active-model.
    - Tabular: POST /predict/tabular with comma-separated numbers; uses active model.
    - Image: POST /predict/image; heuristic response by default.
    - Each has an always-open “Response” pre block for JSON.
- Request handling (UI):
  - fetch wrapper shows a single global status (Idle/OK/Failed) with method/path and reason (error/message field or raw text, or “Timed out after 20s”).
  - Buttons show “Processing…” while requests run; responses print to dedicated <pre> blocks.
  - Client timeout 20s, with clear reason messages.

Docker Compose
- ai-service:
  - Runs python app.py (fast startup, no dev reloader), mounts ./models:/app/models.
  - Env: MODELS_DIR, MODEL_REGISTRY_PATH, ACTIVE_MODEL_NAME, USE_FILE_MODELS="false".
- web-app:
  - Runs flask --reload with ./web-app bind-mount (hot reload for UI only).
- Healthchecks enabled for both.

Models directory
- Host ./models (mounted to /app/models).
- Registry file ./models/model_registry.json contains compact entries including runtime_supported.
- Demo files exist (metadata-only): demo-stats-model.pkl, demo-vision.safetensors, demo-llm.gguf.
- tiny-iris-logreg.joblib remains the executable demo model (tabular) and is the active model by default.

Behavior summary
- “Metadata” endpoints: health, model list, upload, select.
- “Uses model” endpoints: tabular predict, image predict, XAI (always operate on the active model).
- With USE_FILE_MODELS=false:
  - Tabular predictions use DummyTabularModel (instant, deterministic).
  - Image predictions return heuristic metrics/grade; no file model load.
- With USE_FILE_MODELS=true:
  - Tabular loads .joblib/.pkl via joblib; must activate a runtime_supported model first.
  - Image loader is a stub to implement (add real loaders in image_pipeline.load_model).

Key improvements made
- Built single-page UI with two cards and global status bar; responses always visible.
- Clear endpoint paths and chips: “metadata” vs “uses model.”
- Robust proxying in web-app (no 500 from JSON decoding errors).
- Fixed tabular predict_proba list handling.
- Simplified, compact model registry JSON with ready-to-use entries.

Good places for GPT to extend
- Implement real image model loader in image_pipeline.load_model(), and wire model usage into analyse_image().
- Add model-specific schema (e.g., classes/labels) to registry and surface in UI.
- Add per-endpoint latency and basic analytics in UI.
- Provide optional per-request model_name override (power users) while retaining active model default.

