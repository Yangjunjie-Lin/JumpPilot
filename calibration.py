"""Resolution-normalized distance -> duration model, with strict validation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from vision import REFERENCE_WIDTH


@dataclass(frozen=True)
class PressModel:
    k: float = 2.70
    b: float = 0.0
    gain: float = 1.0
    minimum_ms: int = 80
    maximum_ms: int = 1800

    def validate(self) -> None:
        values = (self.k, self.b, self.gain, self.minimum_ms, self.maximum_ms)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("参数必须是有限数字。")
        if not .1 <= self.k <= 12:
            raise ValueError("k 必须在 0.1–12 ms/标准像素之间。")
        if not -500 <= self.b <= 500:
            raise ValueError("b 必须在 -500–500 ms 之间。")
        if not .5 <= self.gain <= 1.5:
            raise ValueError("时长微调系数必须在 0.5–1.5 之间。")
        if not 40 <= self.minimum_ms < self.maximum_ms <= 2500:
            raise ValueError("安全时长范围必须在 40–2500 ms 内。")

    def predict(self, distance_px: float, image_width: int) -> int:
        self.validate()
        if not math.isfinite(distance_px) or distance_px <= 0 or image_width <= 0:
            raise ValueError("距离和画面宽度必须大于零。")
        normalized = distance_px * REFERENCE_WIDTH / image_width
        value = self.gain * (self.k * normalized + self.b)
        if not self.minimum_ms <= value <= self.maximum_ms:
            raise ValueError(f"计算时长 {value:.0f} ms 超过安全范围，已禁止按压；请重新校准。")
        return round(value)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Sample:
    distance_px: float
    width: int
    duration_ms: float

    def validate(self) -> None:
        if not all(math.isfinite(v) for v in (self.distance_px, self.width, self.duration_ms)):
            raise ValueError("样本不能含 NaN 或无穷大。")
        if not 1 <= self.distance_px <= 5000 or not 150 <= self.width <= 6000:
            raise ValueError("样本距离或截图宽度无效。")
        if not 40 <= self.duration_ms <= 2500:
            raise ValueError("样本按压时长必须在 40–2500 ms 之间。")


def fit_model(samples: list[Sample]) -> tuple[PressModel, float]:
    if len(samples) < 2:
        raise ValueError("至少需要两条距离不同、已确认落在中心附近的成功样本。")
    for sample in samples:
        sample.validate()
    x = np.array([s.distance_px * REFERENCE_WIDTH / s.width for s in samples])
    y = np.array([s.duration_ms for s in samples])
    if np.ptp(x) < 30:
        raise ValueError("样本距离太相近；请采集相差至少 30 标准像素的成功跳跃。")
    k, b = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
    model = PressModel(k=float(k), b=float(b))
    model.validate()
    rmse = float(np.sqrt(np.mean((k * x + b - y) ** 2)))
    if rmse > 100:
        raise ValueError(f"样本偏差过大（RMSE {rmse:.1f} ms），请检查距离和时长是否对应。")
    return model, rmse
