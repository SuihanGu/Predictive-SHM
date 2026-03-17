## 🏯 **Multi-Source Data Fusion and Hybrid Deep Learning for Structural Health Monitoring: Application to Yuhuangge Pavilion**

*A real-time visualization platform for multi-sensor structural health monitoring, built with Vue3 + TypeScript + Vite + ECharts.*

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

**从仓库根目录（Monorepo 推荐）：**

```bash
pnpm install          # 安装前端依赖（workspace）
pnpm dev              # 同时启动前端 + 后端（推荐）
pnpm dev:frontend     # 仅启动前端 → http://localhost:3000
pnpm dev:backend      # 仅启动后端 → http://localhost:4999
pnpm build            # 构建前端
```

> ⚠️ **避免 ECONNREFUSED**：前端需代理 `/api` 到后端 4999 端口，请用 `pnpm dev` 同时启动前后端，或先单独运行 `pnpm dev:backend`。

**或分别进入子目录：**

- **前端：** `cd frontend && pnpm install && pnpm dev` → http://localhost:3000  
- **后端：** `cd backend && pip install -r requirements.txt && python run.py` → http://localhost:4999  

**Docker：** `docker-compose up -d`

### Requirements

* Node.js ≥ 18, pnpm ≥ 8（前端）
* Python ≥ 3.10（后端）

---

## 📂 Project Structure (Monorepo)

本仓库采用 **Monorepo** 架构，根目录通过 `pnpm-workspace.yaml` 管理前端工作区，后端为独立 Python 应用。

```
predictive-shm/
├── package.json              # 根脚本：pnpm dev / pnpm dev:frontend / pnpm dev:backend
├── pnpm-workspace.yaml       # 工作区：frontend
├── frontend/                  # 工作区包：Vue3 + Vite
│   ├── src/views/Monitor.vue, main.ts, App.vue, ...
│   ├── index.html, package.json, vite.config.ts, route.ts
│   └── Dockerfile
├── backend/                     # Flask（论文 §2.1）
│   ├── app/ (core, adapters, services, routers), main.py
│   ├── models/, scripts/, sample_data/
│   ├── prediction_service.py, requirements.txt, run.py
│   └── Dockerfile
├── docs/, data_2025_08_to_now/, image/
├── docker-compose.yml, README.md, LICENSE
```

---

## 📡 传感器接入

**真实传感器接入**：详见 [docs/REAL_SENSOR_INTEGRATION.md](docs/REAL_SENSOR_INTEGRATION.md)

支持方式：HTTP API 上报、CSV 文件、桥接脚本（数据库/Modbus/MQTT）。

---

## Supported Sensors（示例）

### 1. Crack Meters

* Device IDs: 623622, 623628, 623641
* Data Field: `data1`

### 2. Tilt Sensors

**X-direction:** 00476464, 00476465, 00476466, 00476467
**Y-direction:** same devices

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

## 📊 Visualization Examples


### System Architecture Diagram

![System Architecture](./docs/system-architecture.png)

> Diagram illustrating data acquisition, processing, model inference, and visualization layers

### Case Study Dashboard (e.g., Yu Huang Ge Temple)

![Case Study Dashboard](./docs/example-dashboard.png)

> Real deployment example with multi-sensor data visualization and predictive results

### Multi-Sensor Line Chart

![Multi-Sensor Line Chart](./docs/example-linechart.png)

> Example of multi-device trend visualization

---

## ⚙️ Development

### Refresh Interval

```ts
const REFRESH_INTERVAL = 10 * 60 * 1000; // 10 min
```

### Time Range

```ts
const dayAgo = now - 24 * 60 * 60; // 24 hours
```

### API Proxy for CORS

```ts
server: {
  proxy: {
    '/api': {
      target: 'my api',
      changeOrigin: true,
      rewrite: path => path.replace(/^\/api/, '')
    }
  }
}
```

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
