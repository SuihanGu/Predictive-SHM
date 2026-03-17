import { spawn } from "node:child_process";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:4999/";
const BACKEND_TIMEOUT_MS = Number(process.env.BACKEND_TIMEOUT_MS || 60_000);
const POLL_MS = Number(process.env.BACKEND_POLL_MS || 800);

function log(msg) {
  process.stdout.write(`${msg}\n`);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForBackend(url, timeoutMs) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url, { method: "GET" });
      if (res.ok) return true;
    } catch {
      // ignore until timeout
    }
    await sleep(POLL_MS);
  }
  return false;
}

function spawnCmd(command, args, opts = {}) {
  const child = spawn(command, args, {
    stdio: "inherit",
    shell: true,
    ...opts,
  });
  return child;
}

let backend = null;
let frontend = null;

function shutdown(code = 0) {
  if (frontend && !frontend.killed) frontend.kill();
  if (backend && !backend.killed) backend.kill();
  process.exit(code);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

// 若后端已在运行（例如用户在另一个终端已启动），则不重复拉起，避免端口冲突
const alreadyUp = await waitForBackend(BACKEND_URL, 1200);
if (alreadyUp) {
  log("[dev] 检测到后端已就绪，跳过启动后端。");
} else {
  log("[dev] 启动后端…");
  backend = spawnCmd("pnpm", ["run", "dev:backend"]);
}

const backendUp = await waitForBackend(BACKEND_URL, BACKEND_TIMEOUT_MS);
if (!backendUp) {
  log(`[dev] 后端在 ${BACKEND_TIMEOUT_MS}ms 内未就绪：${BACKEND_URL}`);
  shutdown(1);
}

log("[dev] 后端已就绪，启动前端…");
frontend = spawnCmd("pnpm", ["run", "dev:frontend"]);

if (backend) {
  backend.on("exit", (code) => {
    log(`[dev] 后端退出 (code=${code ?? "null"})，停止前端…`);
    shutdown(code ?? 1);
  });
}

frontend.on("exit", (code) => {
  log(`[dev] 前端退出 (code=${code ?? "null"})，停止后端…`);
  shutdown(code ?? 0);
});

