# Quicker Web App

This folder contains a separated web wrapper around the existing Quicker
retrieval and reasoning pipeline.

## Start

From the project root:

```bash
conda run -n quicker python web/backend/server.py
```

Open:

```text
http://127.0.0.1:8765
```

Optional environment variables:

```bash
QUICKER_WEB_HOST=127.0.0.1
QUICKER_WEB_PORT=8765
QUICKER_PHASE_PYTHON="conda run -n quicker python"
QUICKER_RETRIEVAL_DEVICE=cpu
```

## Structure

- `backend/server.py`: JSON API, task state machine, Phase subprocess wrapper.
- `frontend/index.html`: UI shell.
- `frontend/styles.css`: UI styling.
- `frontend/app.js`: client-side API calls, polling, editing, uploading.
- `runtime/`: generated task status, task configs, and unmatched uploads.

## Flow

1. Select disease and submit a clinical question.
2. The backend runs EvidenceQA retrieval and LLM judge.
3. If the knowledge base can answer, the UI shows the answer directly.
4. Otherwise, a reasoning task is created.
5. Phase1 pauses for PICO review and editing.
6. Phase3 record screening pauses for PDF upload.
7. Later phases run automatically.
8. Existing local outputs are reused by the Phase scripts when available.
