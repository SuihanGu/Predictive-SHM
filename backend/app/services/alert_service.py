"""
预警服务：完全解耦于预测模型，仅接收 (history, prediction) 做阈值判断。
策略：静态阈值 + 预测残差，两种独立检查。
"""
from typing import Dict, List, Any, Optional
import numpy as np
from app.services.config_loader import get_sensors


DEFAULT_SENSOR_KEYS = ["crack", "tilt_x", "tilt_y", "settlement", "water_level"]

# 默认值：(static, residual)
_DEFAULT_PAIRS = {
    "crack": (0.8, 0.1),
    "tilt_x": (0.5, 0.05),
    "tilt_y": (0.5, 0.05),
    "settlement": (5.0, 0.5),
    "water_level": (100.0, 10.0),
}


def _initial_thresholds_from_config() -> Dict[str, Dict[str, float]]:
    """从 config 加载默认：static（实测值上限）+ residual（预测残差）"""
    out: Dict[str, Dict[str, float]] = {}
    for k, (s_default, r_default) in _DEFAULT_PAIRS.items():
        out[k] = {"static": float(s_default), "residual": float(r_default)}
    for s in get_sensors():
        k = s.get("key")
        if not k:
            continue
        static = s.get("default_static_threshold")
        residual = s.get("default_residual_threshold")
        if k not in out:
            out[k] = {"static": 0.0, "residual": 0.0}
        if static is not None:
            out[k]["static"] = float(static)
        if residual is not None:
            out[k]["residual"] = float(residual)
        # 兼容旧 config：仅有 default_threshold 时用于 residual
        if residual is None and s.get("default_threshold") is not None:
            out[k]["residual"] = float(s["default_threshold"])
    return out


class AlertService:
    """预警逻辑独立于预测模型，仅基于数值与阈值判断"""

    def __init__(self):
        self.thresholds: Dict[str, Dict[str, float]] = _initial_thresholds_from_config()

    def set_thresholds(self, new_thresholds: Dict[str, Any]):
        """支持 { key: { static, residual } } 或旧格式 { key: number } 兼容"""
        if not new_thresholds:
            return
        for k, v in new_thresholds.items():
            if k not in self.thresholds:
                self.thresholds[k] = {"static": 0.0, "residual": 0.0}
            if isinstance(v, (int, float)):
                self.thresholds[k]["residual"] = float(v)
            elif isinstance(v, dict):
                if "static" in v and v["static"] is not None:
                    self.thresholds[k]["static"] = float(v["static"])
                if "residual" in v and v["residual"] is not None:
                    self.thresholds[k]["residual"] = float(v["residual"])

    def check_alerts(
        self,
        history: np.ndarray,
        prediction: np.ndarray,
        sensor_keys: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        双策略、完全解耦：
        - 静态阈值：实测值超过上限 → 告警
        - 预测残差：|预测 - 实测| 超过阈值 → 告警（捕捉趋势异常）
        """
        alerts: List[Dict[str, Any]] = []
        keys = sensor_keys or DEFAULT_SENSOR_KEYS

        h = np.asarray(history)
        p = np.atleast_1d(prediction)
        if p.ndim > 1:
            p = p.flatten()
        if h.ndim == 2:
            h = h[np.newaxis]
        elif h.ndim == 1:
            h = h.reshape(1, -1, 1)
        n = min(h.shape[2] if h.ndim >= 3 else h.shape[1], p.shape[0], len(keys))

        for i in range(n):
            if h.ndim >= 3 and h.shape[2] > i:
                last_val = float(h[0, -1, i])
            elif h.ndim == 2 and h.shape[1] > i:
                last_val = float(h[-1, i])
            else:
                last_val = 0.0
            pred_val = float(p[i]) if i < p.size else 0.0
            name = keys[i] if i < len(keys) else f"sensor_{i}"
            cfg = self.thresholds.get(name) or {"static": 0.0, "residual": 0.0}
            static_th = cfg.get("static") or 0.0
            residual_th = cfg.get("residual") or 0.0

            # 1. 静态阈值：实测值超过上限
            if static_th > 0 and last_val > static_th:
                alerts.append({
                    "type": name,
                    "level": "warning",
                    "rule": "static",
                    "message": f"{name} 实测值 {last_val:.3f} 超过安全限值 {static_th}",
                })

            # 2. 预测残差：模型预测与实测偏差过大
            if residual_th > 0:
                resid = abs(pred_val - last_val)
                if resid > residual_th:
                    alerts.append({
                        "type": name,
                        "level": "warning",
                        "rule": "residual",
                        "message": f"{name} 预测残差 |{pred_val - last_val:.3f}| 超过阈值 {residual_th}",
                    })
        return alerts
