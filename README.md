## 🏯 **Multi-Source Data Fusion and Hybrid Deep Learning for Structural Health Monitoring: Application to Yuhuangge Pavilion**

*A real-time visualization platform for multi-sensor Structural Health Monitoring (SHM), built with Vue 3 + TypeScript + Vite + ECharts, with a Python backend for data access and prediction.*

---

## 📘 Project Overview

The **SHM-System** is a web-based platform for structural health monitoring developed based on a Transformer Encoder-CNN hybrid deep learning model. It visualizes real-time and historical structural health data, including settlement, tilt, crack width, and water level. The system integrates multiple sensors and provides **trend analysis, automatic refresh, and predictive insights** through multi-step ahead predictions (60 minutes). The Transformer Encoder captures long-range temporal dependencies, while 1D-CNN extracts local features, enabling accurate multi-source data fusion and prediction. The system is suitable for historical building preservation, infrastructure monitoring, and research applications.

---

## 🛠️ Technology Stack

| Technology   | Purpose               |
| ------------ | --------------------- |
| Vue 3        | Front-end framework   |
| TypeScript   | Type-safe development |
| Vite         | Modern build tool     |
| Vue Router   | Routing               |
| Element Plus | UI component library  |
| ECharts      | Data visualization    |
| Axios        | HTTP client           |

---

## 🚀 Quick Start

### Run from the repository root (recommended, monorepo)

```bash
pnpm install          # install workspace dependencies
pnpm dev              # start frontend + backend (recommended)
pnpm dev:frontend     # start frontend only  -> http://localhost:3000
pnpm dev:backend      # start backend only   -> http://localhost:4999
pnpm build            # build frontend
```

> If the frontend shows **ECONNREFUSED**, make sure the backend is running on port **4999** and that the frontend proxy routes `/api` to the backend. Using `pnpm dev` is the easiest way to start both.

### Run from subdirectories

- **Frontend**: `cd frontend && pnpm install && pnpm dev` → `http://localhost:3000`  
- **Backend**: `cd backend && pip install -r requirements.txt && python run.py` → `http://localhost:4999`  

### Docker (recommended for a quick demo)

```bash
docker-compose up -d
```

- **Frontend**: `http://localhost:3000` (served by the container on port 80, mapped to 3000)
- **Backend**: `http://localhost:4999`

### Requirements

* Node.js ≥ 18, pnpm ≥ 8 (frontend)
* Python ≥ 3.10 (backend)
* Docker (optional)

---

## 📂 Project Structure (Monorepo)

This repository uses a **monorepo** layout. The root manages the frontend workspace via `pnpm-workspace.yaml`, while the backend is an independent Python service.

```
predictive-shm/
├── package.json               # root scripts: pnpm dev / pnpm dev:frontend / pnpm dev:backend
├── pnpm-workspace.yaml        # workspace: frontend
├── frontend/                  # Vue 3 + Vite app
│   ├── src/ (views/Monitor.vue, App.vue, ...)
│   ├── package.json, vite.config.ts
│   └── Dockerfile
├── backend/                   # Python API service
│   ├── app/ (adapters, services, routers, ...), main.py
│   ├── config/, models/, scripts/, sample_data/
│   ├── requirements.txt, run.py
│   └── Dockerfile
├── docker-compose.yml, README.md, LICENSE
```

---

## 📡 Sensor Integration

For real sensor onboarding, see `docs/REAL_SENSOR_INTEGRATION.md`.

Supported ingestion options typically include: HTTP API uploads, CSV files, and bridge scripts (e.g., database/Modbus/MQTT).

---

## Reproducibility, pre-trained weights, and sample data

**Pre-trained model (crack / Transformer-CNN).** The registry entry `transformer_cnn` in [`backend/models/model_registry.json`](backend/models/model_registry.json) points to **`backend/models/best_crack_model.pth`** plus **`scaler_all.pkl`** and **`scaler_response.pkl`** (see [`backend/models/README.md`](backend/models/README.md)). **When these files are present**, `POST /predict` uses the real **`TransformerCNNAdapter`**; if they are absent (e.g. a minimal clone), the backend **falls back to `MockAdapter`** for smoke tests. Teams publishing this repo typically **ship the three artifacts in-tree**, or provide them via **Git LFS** / a **tagged Release** with download steps linked here.

**Sample sensor data.** **Reference CSVs** for a multi-sensor layout are under [`backend/sample_data/`](backend/sample_data/) (e.g. `crack.csv`, `tilt_x.csv`, `settlement.csv`, `water_level.csv`, merged `training_data.csv`). They support **pipeline testing, UI demo, and re-training workflows**. **Owing to data-use agreements** on the Yuhuangge deployment, we **are not able to publish** the complete raw monitoring archives; we provide **partial / de-sensitized** excerpts so reviewers and readers can still **exercise ingestion, fusion, and prediction APIs** end-to-end together with the published weights.

**“One-click” demo.** From the repo root, `pnpm dev` or `docker-compose up -d` starts the stack; use the monitor UI or the prediction API with `history_data` shaped like the sample CSVs. The public bundle is intended to **reproduce the software behavior**; **point-by-point numerical identity** with every figure computed on restricted onsite data **may not be expected** from the shared sample alone.

---

## Supported Sensors (examples)

### 1. Crack Meters

* Device IDs: 623622, 623628, 623641
* Data Field: `data1`

### 2. Tilt Sensors

- **X-direction**: 00476464, 00476465, 00476466, 00476467  
- **Y-direction**: same devices

* Data Fields: `data1` (X), `data2` (Y)

### 3. Settlement Sensors

* Device IDs: 004521, 004548, 004591, 152947
* Data Field: `data1` (settlement)

### 4. Water Level Gauge

* Dynamic IDs
* Data Field: `data1` (unit: mm)

### Automatic Refresh

* Default interval: **10 minutes**
* Manual refresh supported
* Displays last update time

---

## ⚙️ Development

### Refresh Interval

```ts
const REFRESH_INTERVAL = 10 * 60 * 1000; // 10 min
```

### Time Range

```ts
const dayAgo = now - 24 * 60 * 60 * 1000; // 24 hours
```

### API Proxy for CORS

```ts
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:4999',
      changeOrigin: true,
      rewrite: path => path.replace(/^\/api/, '')
    }
  }
}
```

### Environment Variables (Docker)

- `SHM_API_BASE_URL`: backend upstream base URL (default in `docker-compose.yml` points to `http://139.159.136.213:4999/iem/shm`)
- `VITE_API_BASE`: frontend API base for the backend (default `http://localhost:4999`)

---

## Extending the Prediction Module

In the paper, **Listing 1** shows the **model registration schema** implemented as [`backend/models/model_registry.json`](backend/models/model_registry.json): a **`models`** array of entries, each binding **`id`**, **`adapter`**, weight **`path`**s, I/O hints, and optional **`meta_file`**. That file is the single source of truth the backend loads for **dynamic adapter selection**.

At request time, **`POST /predict`** in [`backend/app/routers/predict.py`](backend/app/routers/predict.py) runs the inference pipeline: historical rows → **ULDM** → **`ModelAdapter.from_uldm` → `predict` → `to_standard_output`** → **`StandardPrediction`** JSON (this execution path is separate from the registry format in **Listing 1**).

### Adapter interface

- Implement a subclass of **`ModelAdapter`** ([`backend/app/adapters/base.py`](backend/app/adapters/base.py)): at minimum **`predict`**, and for this stack typically **`from_uldm`** (ULDM → model tensor) and optionally **`to_standard_output`** (raw array → time-stamped readings).
- Reference implementations live under **[`backend/app/adapters/`](backend/app/adapters/)** (e.g. Transformer-CNN and ONNX wrappers).

### Registration (JSON) and metadata (JSON / YAML)

1. **Runtime registry** — add a model object to [`backend/models/model_registry.json`](backend/models/model_registry.json): **`id`** (API / UI `model_name`), **`adapter`** (factory key, e.g. `TransformerCNNAdapter`, `ONNXAdapter`), artifact **`path`**s, and optional **`meta_file`**. At load time, the registry merges **`meta_file`** when it is **JSON** (see [`backend/app/adapters/registry.py`](backend/app/adapters/registry.py)). Human-readable **YAML** capability examples for documentation and tooling live under [`backend/models/model_meta/`](backend/models/model_meta/) (see that folder’s README).
2. **Monitor UI list** — optional entries under **`models`** in [`backend/config/monitor_config.json`](backend/config/monitor_config.json) control labels/descriptions shown on the monitor page (`config_loader`).

### Dynamic loading

**`get_adapter(model_id)`** in [`backend/app/adapters/registry.py`](backend/app/adapters/registry.py) reads `model_registry.json`, resolves paths relative to the backend root, constructs the matching adapter (with a short-lived instance cache), and falls back to **`MockAdapter`** if the entry or weights are missing. **New adapter class names** require a corresponding branch (or future dynamic import) in that factory—config alone selects among the built-in adapter types.

### Example (Listing 1 — `model_registry.json`)

The repository ships the following shape (see the file for optional `_comment*` fields):

```json
{
  "models": [
    {
      "id": "transformer_cnn",
      "type": "transformer_cnn",
      "label": "Transformer-CNN Crack Forecasting",
      "description": "Time-series forecasting model (PyTorch) trained for crack meters only. Not applicable to other sensor types unless retrained.",
      "adapter": "TransformerCNNAdapter",
      "path": "models/best_crack_model.pth",
      "scaler_path": "models/scaler_all.pkl",
      "response_scaler_path": "models/scaler_response.pkl",
      "target_sensor": "crack",
      "input_dim": 17,
      "output_dim": 3,
      "pred_steps": 6,
      "meta_file": "models/model_meta/transformer_cnn.json"
    }
  ]
}
```

After registration, call **`POST /predict`** with `model_name` set to the entry’s **`id`** and a **`history_data`** series; the service returns **`prediction`** in the standard time-indexed format.

---

## 🧩 Troubleshooting

* **Dependency issues:** Delete `node_modules`, clean cache
* **Port conflict:** Update `vite.config.ts` port
* **Data not loading:** Check API and browser console, set proxy for CORS
* **Charts not rendering:** Check DOM and ECharts initialization

---

## Comparison with open-source SHM tools

**Table 1. Compared systems**

| Project | Reference |
| --- | --- |
| Predictive-SHM | https://github.com/SuihanGu/Predictive-SHM |
| OpenBDLM | https://github.com/CivML-PolyMtl/OpenBDLM |
| pyOMA | https://github.com/simonmarwitz/pyOMA |
| MIDAS-SHM | https://github.com/human-analysis/midas-shm |

**Table 2. Feature comparison**

| Feature | Predictive-SHM | OpenBDLM | pyOMA | MIDAS-SHM |
| --- | --- | --- | --- | --- |
| Primary intent | End-to-end field-style monitoring stack | BDLM-centric long-horizon time-series modelling & anomaly detection for SHM | Operational modal analysis | Damage assessment via mechanics-informed ML |
| Typical stack | Vue 3 + FastAPI, REST/JSON | MATLAB + toolboxes | Python | Python |
| Browser-based monitoring UI | Yes | No | No | No |
| REST/HTTP ingest API (reference release) | Yes | No native HTTP service in core release | No | No |
| Built-in alignment for async sensors | Yes | Partial | N/A | N/A |
| Unified logical schema | Yes (`model_config.json` + ULDM) | No (BDLM state–space formulation, not PS ULDM) | No | No |
| Configuration-driven pluggable forecasters | Yes | No | No | No |
| Threshold / residual alerting in stack | Yes | No | No | No |
| Dedicated time-series DB required | No | No | No | No |
| Open license | MIT | MIT | GPL | Other — confirm per repository / authors |

Table 1 provides a qualitative comparison of representative open-source SHM projects from two perspectives: tool positioning and deployable form. Predictive-SHM (v1.0) targets field-deployment scenarios by offering an end-to-end browser-based monitoring stack that encompasses data ingestion, visualization, multi-step prediction, and lightweight alerting. It uses the ULDM to unify data column ordering and roles, and supports pluggable extension of prediction models through a registry-plus-adapter mechanism. In contrast, OpenBDLM focuses more on BDLM-based probabilistic modeling and anomaly detection (primarily through MATLAB workflows), pyOMA concentrates on operational modal analysis, and MIDAS-SHM centers on mechanics-constrained damage assessment research workflows. It should be emphasized that the entries in the table (e.g., “lightweight deployment,” “partial support”) represent qualitative judgments: Predictive-SHM, in its reference release, deliberately avoids mandating dedicated time-series databases or clusters so as to lower the operational barrier for demonstrations and reproducibility experiments. Other projects, while they may not require server clusters, may still incur different forms of dependency costs related to desktop runtime environments, commercial runtimes (e.g., MATLAB), or computational resources (e.g., GPUs).

---

## 📄 License

MIT License
Copyright (c) 2025 Siran Yang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---
