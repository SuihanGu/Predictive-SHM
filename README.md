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

## 🧩 Troubleshooting

* **Dependency issues:** Delete `node_modules`, clean cache
* **Port conflict:** Update `vite.config.ts` port
* **Data not loading:** Check API and browser console, set proxy for CORS
* **Charts not rendering:** Check DOM and ECharts initialization

---

## 📄 License

MIT License
Copyright (c) 2025 Siran Yang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---
