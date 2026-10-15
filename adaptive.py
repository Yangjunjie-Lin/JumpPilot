"""Bounded, score-gated landing feedback. No mouse input or persistence here."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math

import cv2
import numpy as np
from PIL import Image

from calibration import PressModel
from vision import Detection, REFERENCE_WIDTH, _resample_contour


@dataclass(frozen=True)
class LandingFeedback:
    accepted: bool
    reason: str
    score_before: int | None = None
    score_after: int | None = None
    along_error: float | None = None  # Standard pixels; positive = too far.
    lateral_error: float | None = None
    confidence: float = 0.
    landed_center: tuple[float, float] | None = None
    distance: float = 0.


def track_landed_platform(after: Image.Image, before: Detection,
                          current: Detection, before_image: Image.Image | None = None) -> tuple[tuple[float, float], float] | None:
    """Match the OLD target outline near the NEW foot, allowing camera panning.

    Contour-to-edge distances ignore the purple piece occlusion. A second
    plausible match, missing edges, a clipped face or a foot outside the face
    rejects the measurement. No raw before/after screen-coordinate subtraction.
    """
    if len(before.polygon) < 4 or (before.width, before.height) != after.size:
        return None
    scale = REFERENCE_WIDTH/after.width
    size = (int(REFERENCE_WIDTH), round(after.height*scale))
    rgb = np.array(after.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    piece_mask = cv2.inRange(hsv, (105, 35, 28), (151, 235, 175))
    piece_mask = cv2.dilate(piece_mask, np.ones((9, 9), np.uint8)) > 0
    edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 20, 55)
    edges[piece_mask] = 0
    distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
    polygon = np.asarray(before.polygon, np.float32)*scale
    try:
        offsets = _resample_contour(polygon[:, None, :], 120)-np.asarray(before.target)*scale
    except ValueError:
        return None
    if before_image is not None:
        if before_image.size != after.size:
            return None
        # Flat-region rhombi are eroded several pixels inside the real rim.
        # Recover each contour sample's real edge in the PRE-JUMP image first;
        # otherwise partial piece occlusion can bias an inset contour sideways.
        previous_rgb = np.array(before_image.convert("RGB").resize(size, Image.Resampling.LANCZOS))
        previous_edges = cv2.Canny(cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY), 20, 55)
        previous_distance = cv2.distanceTransform(255-previous_edges, cv2.DIST_L2, 3)
        tangent = np.roll(offsets, -1, axis=0)-np.roll(offsets, 1, axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-6)
        normal[(normal*offsets).sum(axis=1) < 0] *= -1
        shifts = np.arange(-2, 9.5, .5)
        choices = np.rint(offsets[:, None, :]+np.asarray(before.target)*scale
                          +normal[:, None, :]*shifts[None, :, None]).astype(int)
        xx, yy = np.clip(choices[..., 0], 0, size[0]-1), np.clip(choices[..., 1], 0, size[1]-1)
        costs = previous_distance[yy, xx]+.04*np.abs(shifts)[None, :]
        winners = np.argmin(costs, axis=1)
        selected_distance = previous_distance[yy[np.arange(len(offsets)), winners], xx[np.arange(len(offsets)), winners]]
        if np.mean(selected_distance < 2) < .9:
            return None
        offsets = choices[np.arange(len(offsets)), winners]-np.asarray(before.target)*scale
    elif len(before.polygon) == 4:
        return None  # Do not learn from an inset rhombus without its source image.
    foot = np.asarray(current.piece)*scale
    xs = np.arange(round(foot[0])-90, round(foot[0])+91, 2)
    ys = np.arange(round(foot[1])-66, round(foot[1])+67, 2)
    centers = np.column_stack([v.ravel() for v in np.meshgrid(xs, ys)])
    outline = polygon-np.asarray(before.target)*scale
    inside = np.array([cv2.pointPolygonTest(outline.astype(np.float32),
                                           tuple(map(float, foot-center)), True) >= 1 for center in centers])
    centers = centers[inside]
    if len(centers) < 2:
        return None
    points = np.rint(centers[:, None, :]+offsets).astype(int)
    valid = ((points[..., 0] > 1) & (points[..., 0] < size[0]-2)
             & (points[..., 1] > size[1]*.20) & (points[..., 1] < size[1]*.88))
    complete = np.all(valid, axis=1)
    px = np.clip(points[..., 0], 0, size[0]-1)
    py = np.clip(points[..., 1], 0, size[1]-1)
    valid &= ~piece_mask[py, px]
    visible = valid.sum(axis=1)
    residuals = distance[py, px]
    losses = (np.minimum(residuals, 12)*valid).sum(axis=1)/np.maximum(visible, 1)
    losses[~complete | (visible < len(offsets)*.60)] = np.inf
    best = int(np.argmin(losses))
    if not math.isfinite(float(losses[best])):
        return None
    support = float(((residuals[best] <= 3)*valid[best]).sum()/max(visible[best], 1))
    competitors = losses[np.linalg.norm(centers-centers[best], axis=1) > 14]
    margin = float(np.min(competitors)-losses[best]) if len(competitors) else math.inf
    if support < .80 or losses[best] > 2.1 or margin < .9:
        return None
    if before_image is not None:
        interior = np.vstack((offsets*.45, offsets*.70))
        old_points = np.rint(interior+np.asarray(before.target)*scale).astype(int)
        new_points = np.rint(interior+centers[best]).astype(int)
        ox, oy = old_points[:, 0], old_points[:, 1]
        nx, ny = new_points[:, 0], new_points[:, 1]
        valid_colors = ((ox >= 0) & (ox < size[0]) & (oy >= 0) & (oy < size[1])
                        & (nx >= 0) & (nx < size[0]) & (ny >= 0) & (ny < size[1]))
        ox, oy, nx, ny = (values[valid_colors] for values in (ox, oy, nx, ny))
        visible_colors = ~piece_mask[ny, nx]
        if visible_colors.sum() < len(interior)*.50:
            return None
        color_error = np.max(np.abs(previous_rgb[oy, ox].astype(float)-rgb[ny, nx]), axis=1)[visible_colors]
        if np.median(color_error) > 18 or np.mean(color_error < 30) < .75:
            return None  # Same shape alone is not sufficient platform identity.
    confidence = min(.97, .55+.35*support+.07*min(margin/2, 1))
    return tuple(map(float, centers[best]/scale)), confidence


def measure_landing(after: Image.Image, before: Detection, current: Detection,
                    score_before: int | None, score_after: int | None,
                    duration: int, actual_ms: float, before_image: Image.Image | None = None) -> LandingFeedback:
    base = LandingFeedback(False, "", score_before, score_after, distance=before.normalized_distance)
    def reject(reason):
        return replace(base, reason=reason)
    if score_before is None or score_after is None:
        return reject("分数未稳定识别，本跳不学习")
    if not 0 < score_after-score_before <= 100:
        return reject("未检测到可信加分，本跳不学习")
    if before.manual or current.manual or min(before.confidence, current.confidence) < .84:
        return reject("点位来自人工修改或算法评分不足，本跳不学习")
    if not math.isfinite(actual_ms) or abs(actual_ms-duration) > max(40, duration*.08):
        return reject("实际按压时长波动较大，本跳不学习")
    tracked = track_landed_platform(after, before, current, before_image)
    if tracked is None:
        return reject("无法唯一匹配本次落地平台，本跳不学习")
    center, confidence = tracked
    direction = np.array((before.dx, before.dy))/max(before.distance, 1)
    error = (np.asarray(current.piece)-center)*REFERENCE_WIDTH/before.width
    along = float(error @ direction)
    lateral = float(error @ np.array((-direction[1], direction[0])))
    base = replace(base, along_error=along, lateral_error=lateral, confidence=confidence, landed_center=center)
    if abs(lateral) > 12 or abs(along) > min(60, before.normalized_distance*.25):
        return replace(base, reason="落点横向偏差或偏离过大，请人工检查，不自动学习")
    label = "接近中心" if abs(along) <= 4 else ("偏远" if along > 0 else "偏短")
    return replace(base, accepted=True, reason=f"得分 +{score_after-score_before}，落点{label} {abs(along):.1f} 标准像素")


class AdaptiveLearner:
    """Three consistent trusted observations before a <=2% gain change.

    k/b stay fixed: a single flight cannot identify both slope and intercept.
    The session gain remains within +/-15% of the explicitly supplied baseline.
    Old/failed/conflicting observations never carry through to a new batch.
    """
    def __init__(self, model: PressModel):
        model.validate()
        self.baseline = model
        self.model = model
        self._batch = deque(maxlen=3)
        self.accepted = 0
        self.updates = 0

    def observe(self, feedback: LandingFeedback) -> tuple[PressModel, str, bool]:
        if (not feedback.accepted or feedback.along_error is None
                or not math.isfinite(feedback.along_error) or not math.isfinite(feedback.confidence) or feedback.confidence < .84
                or not 45 <= feedback.distance <= 520):
            self._batch.clear()
            return self.model, feedback.reason, False
        self.accepted += 1
        error = feedback.along_error
        if abs(error) <= 4:
            self._batch.clear()
            return self.model, feedback.reason+"；中心死区内，保持参数", False
        if abs(error) > min(60, feedback.distance*.25):
            self._batch.clear()
            return self.model, "偏差过大，保持参数并等待人工检查", False
        ratio = feedback.distance/(feedback.distance+error)
        if self._batch and (ratio-1)*(self._batch[-1]-1) <= 0:
            self._batch.clear()
        self._batch.append(ratio)
        if len(self._batch) < 3:
            return self.model, feedback.reason+f"；一致反馈 {len(self._batch)}/3，暂不改参", False
        delta = float(np.clip((float(np.median(self._batch))-1)*.35, -.02, .02))
        self._batch.clear()
        new_gain = float(np.clip(self.model.gain*(1+delta),
                                 max(.5, self.baseline.gain*.85), min(1.5, self.baseline.gain*1.15)))
        if abs(new_gain-self.model.gain) < .0001:
            return self.model, "达到本轮学习边界，保持参数；需要重新人工校准", False
        old_gain = self.model.gain
        self.model = replace(self.model, gain=new_gain)
        self.model.validate()
        self.updates += 1
        return self.model, feedback.reason+f"；时长倍率 {old_gain:.4f} → {new_gain:.4f}（k/b 不变）", True
