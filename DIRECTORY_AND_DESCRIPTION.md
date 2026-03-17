# Predictive-SHM 当前系统目录及说明

本文档描述项目 **Predictive-SHM**（结构健康监测与预测系统）的目录结构及各文件/目录的用途说明。  
**已按 Monorepo 重构**：前端在 `frontend/`，后端在 `backend/`。  
*生成日期：2025-03-01；重构后更新*

---

## 一、根目录总览（当前结构）

```
Predictive-SHM/
├── package.json              # 根 package（Monorepo 脚本）
├── pnpm-workspace.yaml       # pnpm 工作区：frontend
├── frontend/                    # Vue3 + Vite 前端（原根目录前端整体迁入）
│   ├── index.html, package.json, vite.config.ts, route.ts
│   ├── src/
│   │   ├── index.vue, main.ts, App.vue, style.css, vite-env.d.ts
│   │   └── views/Monitor.vue
│   └── Dockerfile
├── backend/                     # Flask 后端（论文 §2.1 架构）
│   ├── app/
│   │   ├── core/                # config.py, dependencies.py
│   │   ├── adapters/            # model_adapter.py（§2.3）
│   │   ├── services/            # data_processor.py（§2.2）, alert_service.py（§2.4）
│   │   ├── routers/             # data, predict, alerts, models
│   │   └── main.py
│   ├── models/                  # best_crack_model.pth, scaler_*.pkl, train_crack_model.py
│   ├── scripts/                 # download_*.py, prepare_training_data.py 等
│   ├── sample_data/             # 示例数据（README）
│   ├── prediction_service.py    # 原有预测服务（保留）
│   ├── requirements.txt, run.py, Dockerfile
├── docs/                        # 根目录保留
├── data_2025_08_to_now/        # 根目录保留
├── image/                       # 根目录保留
├── docker-compose.yml, README.md, LICENSE, DIRECTORY_AND_DESCRIPTION.md
└── node_modules/                # 仅 frontend 需 pnpm install（在 frontend/ 下）
```

---

## 二、根目录文件说明

| 文件 | 说明 |
|------|------|
| **index.html** | 前端单页应用入口 HTML，挂载 `#app`，引用 `/src/main.ts`。标题：Structural Health Monitoring System。 |
| **index.vue** | 根级 Vue 组件，被 `route.ts` 中 SHM 监测页路由引用（`../../SHM/index.vue`），可能作为监测视图的入口或布局。 |
| **package.json** | 前端项目配置。项目名 `shm-monitor`，描述为「结构健康监测数据可视化系统」。脚本：`dev`（Vite）、`build`、`preview`、`lint`。依赖：Vue3、Vue Router、ECharts、Element Plus、Axios。 |
| **pnpm-lock.yaml** | pnpm 依赖锁定文件。 |
| **tsconfig.json** / **tsconfig.node.json** | TypeScript 配置（含 Node 环境配置）。 |
| **vite.config.ts** | Vite 配置：`@` → `src`、端口 3000、`/api` 代理到 `http://139.159.136.213:4999`，构建输出 `dist`。 |
| **route.ts** | 路由配置片段：`/shm` 下挂载 Layout，重定向到 `/shm/monitor`，子路由引用 `../../SHM/index.vue` 作为「监测数据」页。 |
| **prediction_service.py** | **预测服务主程序**：Flask + Transformer-CNN 裂纹预测。拉取数据、加载 `models/` 下权重与 scaler、定时预测、提供 HTTP 接口。 |
| **requirements.txt** | Python 依赖：flask、flask-cors、torch、numpy、pandas、scikit-learn、requests。 |
| **download_data.py** | 数据下载脚本（具体数据源见脚本内）。 |
| **download_data_from_2024_08.py** | 自 2024 年 8 月起的数据下载脚本。 |
| **extract_temperature.py** | 温度数据提取/处理脚本。 |
| **fill_missing_values.py** | 缺失值填充脚本。 |
| **prepare_training_data.py** | 训练数据预处理脚本。 |
| **start_prediction_service.bat** | Windows 下启动预测服务脚本。 |
| **start_prediction_service.sh** | Linux/macOS 下启动预测服务脚本。 |
| **README.md** | 项目说明：SHM-System、技术栈（Vue3/TS/Vite/ECharts）、传感器类型、刷新间隔、开发与排错说明、MIT 许可。 |

---

## 三、src/（前端源码）

| 路径 | 说明 |
|------|------|
| **src/main.ts** | 前端入口，挂载 Vue 应用。 |
| **src/App.vue** | 根组件。 |
| **src/style.css** | 全局样式。 |
| **src/vite-env.d.ts** | Vite 环境类型声明。 |
| **src/views/Monitor.vue** | 核心监测页面（仪表盘、图表、多传感器数据与预测展示）。 |

说明：当前前端为单应用结构，路由通过 `route.ts` 指向 `../../SHM/index.vue`，与 `src/views/Monitor.vue` 的关系需在工程内确认（是否同一功能或不同入口）。

---

## 四、models/（模型与预处理）

| 路径 | 说明 |
|------|------|
| **models/best_crack_model.pth** | 裂纹预测 Transformer-CNN 模型权重（PyTorch）。 |
| **models/scaler_all.pkl** | 全量特征标准化器（MinMaxScaler 等）序列化文件。 |
| **models/scaler_response.pkl** | 响应变量标准化器。 |
| **models/train_crack_model.py** | 裂纹模型训练脚本（与 prediction_service 中结构一致）。 |

---

## 五、data_2025_08_to_now/ 与压缩包

| 路径 | 说明 |
|------|------|
| **data_2025_08_to_now.7z** | 2025 年 8 月至今数据的压缩包。 |
| **data_2025_08_to_now/** | 解压后的 CSV 数据目录，内含多传感器 CSV（如水位、水准点、斜测探头、裂缝等，文件名可能为中文编码）。 |

---

## 六、docs/ 与 image/

| 路径 | 说明 |
|------|------|
| **docs/system-architecture.png** | 系统架构示意图。 |
| **docs/example-dashboard.png** | 示例仪表盘截图（如玉皇阁案例）。 |
| **docs/example-linechart.png** | 多传感器折线图示例。 |
| **image/yuhuang/image.png** | 玉皇阁相关图片资源。 |

---

## 七、未出现的目录/文件（与常见 Monorepo 对比）

| 项目 | 当前状态 |
|------|----------|
| **backend/** | 无；Python 服务与脚本在根目录。 |
| **frontend/** | 无；前端直接在根目录（src/、vite 等）。 |
| **docker-compose.yml** | 无。 |
| **LICENSE** | 根目录无文件；README 中声明为 MIT。 |
| **backend/app/**（FastAPI） | 无；当前为 Flask 预测服务。 |
| **backend/models/** | 无；模型在根目录 `models/`。 |
| **backend/sample_data/** | 无。 |

---

## 八、技术栈与数据流概要

- **前端**：Vue 3 + TypeScript + Vite + Vue Router + Element Plus + ECharts + Axios；开发端口 3000，`/api` 代理到远程 4999。
- **后端/预测**：Flask + PyTorch Transformer-CNN；从远程 API 拉数据，用 `models/` 下 `.pth` 与 `.pkl` 做多步预测。
- **传感器**：裂缝计、倾角、沉降、水位等；设备 ID 与字段见 README。
- **数据**：`data_2025_08_to_now/` 及脚本用于训练/预处理；预测服务运行时另从线上 API 取数。

---

## 九、若升级为 Monorepo 时的对应关系建议

| 当前位置 | 建议迁入位置（与您之前给出的结构对齐） |
|----------|----------------------------------------|
| 根目录前端（index.html, src/, vite 等） | **frontend/**（整体移入，保持 src/views/Monitor.vue 等） |
| prediction_service.py + requirements.txt + 模型/数据脚本 | **backend/**：FastAPI 重构后为 app/，旧脚本可放 backend/scripts 或保留在根目录做离线用 |
| models/*.pth, *.pkl | **backend/models/** 或保留根目录（按论文开源示例放 backend/sample_data 与 backend/models） |
| data_2025_08_to_now*、image、docs | 根目录或 **backend/sample_data/**（示例数据）、**docs/** 保留在根目录 |
| README.md、LICENSE | 根目录保留；README 已为论文风格可继续沿用 |

本文档仅描述当前系统目录与说明，不修改任何代码或目录结构。  
若需我按您给出的 Monorepo 结构生成迁移步骤或具体文件清单，可说明目标仓库名称（如 predictive-shm）与是否保留现有 Git 历史。
