#!/bin/bash
# 结构迁移到 Monorepo（从仓库根目录执行）
# 若 frontend/、backend/ 已存在则跳过对应 mv，避免覆盖

set -e
cd "$(dirname "$0")"

echo "=== Creating directories ==="
mkdir -p frontend
mkdir -p backend/app/routers backend/app/services backend/app/adapters backend/scripts backend/sample_data

echo "=== Migrating frontend (skip if already in frontend/) ==="
for f in index.html index.vue package.json pnpm-lock.yaml vite.config.ts route.ts; do
  [ -f "$f" ] && mv "$f" frontend/ && echo "  moved $f"
done
for f in tsconfig.json tsconfig.node.json; do
  [ -f "$f" ] && mv "$f" frontend/ && echo "  moved $f"
done
[ -d "src" ] && [ ! -d "frontend/src" ] && mv src frontend/ && echo "  moved src/"

echo "=== Migrating backend ==="
[ -f "prediction_service.py" ] && mv prediction_service.py backend/ && echo "  moved prediction_service.py"
[ -f "requirements.txt" ] && mv requirements.txt backend/ && echo "  moved requirements.txt"
for f in start_prediction_service.bat start_prediction_service.sh; do
  [ -f "$f" ] && mv "$f" backend/ && echo "  moved $f"
done
[ -d "models" ] && [ ! -d "backend/models" ] && mv models backend/ && echo "  moved models/"
for f in data_2025_08_to_now data_2025_08_to_now.7z; do
  [ -e "$f" ] && mv "$f" backend/sample_data/ && echo "  moved $f -> backend/sample_data/"
done
for f in download_*.py extract_*.py fill_*.py prepare_*.py; do
  [ -f "$f" ] && mv "$f" backend/scripts/ && echo "  moved $f -> backend/scripts/"
done

echo "=== Done. Next: cd frontend && pnpm install && pnpm dev; cd backend && pip install -r requirements.txt && python prediction_service.py ==="
