"""Deterministic, local-only vision for the classic purple-piece jump game.

This is deliberately a conservative heuristic, not a general game detector.
Unsupported themes and ambiguous surfaces return an error instead of a click.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from top_geometry import contour_quads, current_ellipses, line_quads, outer_surfaces, piece_occlusion, quad_center

REFERENCE_WIDTH = 540.0


class DetectionError(ValueError):
    pass


@dataclass(frozen=True)
class Detection:
    piece: tuple[float, float]
    target: tuple[float, float]
    confidence: float
    width: int
    height: int
    polygon: tuple[tuple[float, float], ...] = ()
    manual: bool = False
    note: str = ""
    confidence_override: float | None = None
    human_confirmed: bool = False
    current_center: tuple[float, float] | None = None
    current_polygon: tuple[tuple[float, float], ...] = ()
    current_confidence: float = 0.
    target_confidence: float = 0.

    @property
    def center_distance(self) -> float | None:
        """Top centre S -> T, distinct from the actual flight A -> T."""
        return None if self.current_center is None else math.dist(self.current_center, self.target)

    @property
    def effective_confidence(self) -> float:
        """Keep the measured score intact; a user's score is explicitly separate."""
        return self.confidence if self.confidence_override is None else self.confidence_override

    def with_confidence(self, value: float | None) -> Detection:
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("人工置信度必须在 0%–100% 之间。")
        return replace(self, confidence_override=value, human_confirmed=False)

    def confirmed(self) -> Detection:
        return replace(self, human_confirmed=True)

    @property
    def dx(self) -> float:
        return self.target[0] - self.piece[0]

    @property
    def dy(self) -> float:
        return self.target[1] - self.piece[1]

    @property
    def distance(self) -> float:
        return math.hypot(self.dx, self.dy)

    @property
    def normalized_distance(self) -> float:
        return self.distance * REFERENCE_WIDTH / self.width

    def with_point(self, name: str, point: tuple[float, float]) -> Detection:
        if name not in ("piece", "target", "current_center"):
            raise ValueError("Unknown point")
        if not (0 <= point[0] < self.width and 0 <= point[1] < self.height):
            raise ValueError("Point outside image")
        return replace(self, **{name: point}, manual=True,
                       confidence_override=None, human_confirmed=False,
                       polygon=() if name == "target" else self.polygon,
                       current_polygon=() if name in ("piece", "current_center") else self.current_polygon,
                       current_confidence=0. if name in ("piece", "current_center") else self.current_confidence,
                       target_confidence=0. if name == "target" else self.target_confidence,
                       **({"current_center": None} if name == "piece" else {}),
                       note="手动校正；仅允许单步试跳")


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        if image.width * image.height > 24_000_000:
            raise DetectionError("图片过大，请裁剪到游戏区域后重试。")
        return ImageOps.exif_transpose(image).convert("RGB")


def _piece_candidates(rgb: np.ndarray) -> list[tuple[float, tuple[float, float], np.ndarray]]:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (105, 35, 28), (151, 235, 175))
    mask[:int(h * .20)] = 0
    mask[int(h * .87):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = []
    for label in range(1, n):
        x, y, bw, bh, area = map(int, stats[label])
        if not (w * .024 <= bw <= w * .105 and w * .065 <= bh <= w * .26):
            continue
        if not (1.15 <= bh / bw <= 4.8 and area > w * w * .0010):
            continue
        has_head = False
        for j in range(1, n):
            if j == label:
                continue
            hx, hy, hw, hh, ha = map(int, stats[j])
            if (abs(hx + hw / 2 - x - bw / 2) < bw * .32
                    and .38 * bw <= hw <= 1.18 * bw
                    and .6 <= hh / max(hw, 1) <= 1.5
                    and -.1 * bw <= y - hy - hh <= .65 * bw
                    and ha > bw * bw * .12):
                has_head = True
                break
        # Some themes connect the head and the body with a narrow neck.
        ys, xs = np.where(labels == label)
        neck_connected = False
        if bh / bw > 2.2:
            row_widths = np.bincount(ys - y, minlength=bh)
            neck = row_widths[int(bh * .22):max(int(bh * .45), 1)]
            neck_connected = bool(neck.size and neck.min() < bw * .4)
        if not has_head and not neck_connected:
            continue
        bottom = float(np.quantile(ys, .995))
        lower = xs[ys >= bottom - max(3, bw * .22)]
        foot = (float(np.median(lower)), bottom - bw * .22)
        confidence = .94 if has_head else .82
        result.append((confidence, foot, labels == label))
    return sorted(result, key=lambda item: item[0], reverse=True)


def _resample_contour(contour: np.ndarray, count: int = 160) -> np.ndarray:
    """Uniform arc spacing avoids fit bias from CHAIN_APPROX_SIMPLE corners."""
    points = contour.reshape(-1, 2).astype(np.float32)
    points = np.vstack((points, points[0]))
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 1e-5]
    points, lengths = points[keep], lengths[keep]
    if len(points) < 4 or lengths[-1] < 30:
        raise ValueError("Contour too short")
    steps = np.linspace(0, lengths[-1], count, endpoint=False)
    return np.column_stack([np.interp(steps, lengths, points[:, i]) for i in (0, 1)]).astype(np.float32)


def _ellipse_surface(contour: np.ndarray, edge_distance: np.ndarray, source: str) -> dict | None:
    """Fit only complete, near-horizontal top ellipses, never the cylinder body.

    Filled regions preserve an outer rim despite a center hole/texture. Closed
    edge contours provide an independent center, including non-flat top faces.
    Residual, angular coverage and real edge support determine quality.
    """
    h, w = edge_distance.shape
    area = cv2.contourArea(contour)
    x, y, bw, bh = cv2.boundingRect(contour)
    if not (w*w*.0025 < area < w*w*.23 and w*.075 < bw < w*.65):
        return None
    if x <= 2 or y <= h*.20 or x+bw >= w-2 or not 1.25 < bw/max(bh, 1) < 3.5:
        return None
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / max(hull_area, 1)
    if solidity < .91 or len(contour) < 8:
        return None
    try:
        points = _resample_contour(contour)
        (cx, cy), (axis_a, axis_b), degrees = cv2.fitEllipse(points[:, None, :])
    except (cv2.error, ValueError):
        return None
    if min(axis_a, axis_b) < 14:
        return None
    angle = math.radians(degrees)
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    local = (points - (cx, cy)) @ rotation
    normalized = local / (np.array([axis_a, axis_b]) / 2)
    radius = np.linalg.norm(normalized, axis=1)
    residual = np.abs(radius - 1)
    rms = float(np.sqrt(np.mean(residual**2)))
    q90 = float(np.quantile(residual, .90))
    ratio = min(axis_a, axis_b) / max(axis_a, axis_b)
    major_degrees = degrees if axis_a >= axis_b else degrees + 90
    tilt = abs((major_degrees + 90) % 180 - 90)
    area_error = abs(math.pi*axis_a*axis_b/4-area) / max(area, 1)
    if not .29 < ratio < .79 or tilt > 16 or rms > .055 or q90 > .09 or area_error > .14:
        return None
    bins = (np.arctan2(normalized[:, 1], normalized[:, 0]) + math.pi) / (2*math.pi) * 24
    coverage = len(np.unique(np.minimum(23, bins[residual < .07].astype(int)))) / 24
    if coverage < .87:
        return None
    theta = np.linspace(0, 2*math.pi, 96, endpoint=False)
    polygon = np.column_stack((np.cos(theta)*axis_a/2, np.sin(theta)*axis_b/2)) @ rotation.T + (cx, cy)
    xs = np.clip(np.rint(polygon[:, 0]).astype(int), 0, w-1)
    ys = np.clip(np.rint(polygon[:, 1]).astype(int), 0, h-1)
    # Flat regions have already been blurred and eroded. On upscaled small
    # captures their boundary sits up to ~6 pixels inside the visible rim.
    tolerance = 6.5 if source == "flat" else 2.3
    edge_support = float(np.mean(edge_distance[ys, xs] <= tolerance))
    if edge_support < .70:
        return None
    quality = (.38*max(0., 1-rms/.065) + .24*edge_support + .18*coverage
               + .12*max(0., 1-area_error/.16) + .08*max(0., 1-tilt/20))
    score = min(.96, .55 + .42*quality)
    return dict(center=(cx, cy), score=score, polygon=polygon, kind="圆形顶面 · 椭圆几何拟合",
                source=source, diameter=max(axis_a, axis_b), residual=rms, edge_support=edge_support)


def _deduplicate_surfaces(candidates: list[dict]) -> list[dict]:
    """Concentric rims/holes are one platform, not competing next platforms."""
    groups: list[list[dict]] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        for group in groups:
            anchor = group[0]
            if (len(candidate["polygon"]) == len(anchor["polygon"]) == 4
                    and min(candidate["diameter"], anchor["diameter"])/max(candidate["diameter"], anchor["diameter"]) < .75):
                continue  # A small inset quad is not the outer landing face.
            tolerance = max(3.0, min(candidate["diameter"], anchor["diameter"]) * .055)
            if np.linalg.norm(np.subtract(candidate["center"], anchor["center"])) <= tolerance:
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    merged = []
    for group in groups:
        best = group[0].copy()
        trusted = [item for item in group if item["score"] >= best["score"]-.05]
        # Use actual edge fits when available, rather than eroded-region centers.
        edge_fits = [item for item in trusted if item["source"] == "edge"]
        center_sources = edge_fits or trusted
        if all(len(item["polygon"]) == 4 for item in trusted):
            # A perspective centre must belong to the SAME four corners. Do
            # not mix an inset face/side stripe's centre with another polygon.
            exact = [item for item in trusted if item["source"] in ("edge", "rim")]
            exact = exact or trusted
            support = max(item.get("edge_support", 0) for item in exact)
            equally_supported = [item for item in exact if item.get("edge_support", 0) >= support-.02]
            best = max(equally_supported, key=lambda item: item["diameter"]).copy()
        else:
            best["center"] = tuple(np.median([item["center"] for item in center_sources], axis=0))
            best["polygon"] = max(trusted, key=lambda item: item["diameter"])["polygon"]
        if len({item["source"] for item in trusted}) > 1:
            best["score"] = min(.96, best["score"] + .015)
        merged.append(best)
    return sorted(merged, key=lambda item: item["score"], reverse=True)


def _all_surfaces(rgb: np.ndarray, foot: tuple[float, float]) -> list[dict]:
    h, w = rgb.shape[:2]
    # The classic background is a light, gently varying gradient. Estimate it
    # row-by-row, from the lighter of the two image margins.
    margin = max(4, int(w * .035))
    background = np.maximum(np.median(rgb[:, :margin], axis=1),
                            np.median(rgb[:, -margin:], axis=1))
    contrast = np.max(np.abs(rgb.astype(float) - background[:, None, :]), axis=2)
    blur = cv2.GaussianBlur(rgb, (3, 3), 0).astype(np.float32)
    gray = cv2.cvtColor(blur.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 20, 55)
    edge_distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
    dx = np.max(np.abs(blur - np.roll(blur, 1, axis=1)), axis=2)
    dy = np.max(np.abs(blur - np.roll(blur, 1, axis=0)), axis=2)
    flat = ((contrast > 25) & (dx < 4) & (dy < 4)).astype(np.uint8) * 255
    flat[:int(h * .22)] = 0
    flat[min(h, int(foot[1] + w * .25)):] = 0
    flat[:, :2] = flat[:, -2:] = 0
    flat = cv2.erode(flat, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(flat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        circular = _ellipse_surface(contour, edge_distance, "flat")
        if circular is not None:
            candidates.append(circular)
        area = cv2.contourArea(contour)
        if not w * w * .004 < area < w * w * .24:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if not (.09 * w < bw < .65 * w and 1.15 < bw / max(bh, 1) < 3.2):
            continue
        if x <= 2 or x + bw >= w - 2:
            continue  # A clipped platform has no trustworthy visible center.
        m = cv2.moments(contour)
        center = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / max(hull_area, 1)
        extent = area / (bw * bh)
        if solidity < .93:
            continue
        hull = cv2.convexHull(contour)
        poly = cv2.approxPolyDP(hull, .025 * cv2.arcLength(hull, True), True)
        shape = 0.0
        kind = ""
        if len(poly) == 4 and .38 < extent < .65:
            vertices = poly[:, 0, :].astype(float)
            top = vertices[np.argmin(vertices[:, 1])]
            bottom = vertices[np.argmax(vertices[:, 1])]
            left = vertices[np.argmin(vertices[:, 0])]
            right = vertices[np.argmax(vertices[:, 0])]
            # A top rhombus has horizontal/vertical diagonals; side faces do not.
            error = (abs(top[0] - bottom[0]) / bw +
                     abs(left[1] - right[1]) / bh +
                     np.linalg.norm((top + bottom - left - right) / 2) / bw)
            if error < .32:
                shape = .96 - error * .5
                center = quad_center(vertices)
                kind = "菱形顶面"
        if not shape:
            continue
        score = .90 * shape + .10 * min(solidity, 1.0)
        candidates.append(dict(center=center, score=min(score, .99),
                               polygon=poly[:, 0, :], kind=kind, source="rhombus", diameter=bw))
    edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in edge_contours:
        circular = _ellipse_surface(contour, edge_distance, "edge")
        if circular is not None:
            candidates.append(circular)
    candidates.extend(contour_quads(edge_contours, edge_distance))
    occlusion = piece_occlusion(rgb, foot)
    clean_edges = edges.copy()
    clean_edges[occlusion] = 0
    clean_distance = cv2.distanceTransform(255-clean_edges, cv2.DIST_L2, 3)
    candidates.extend(line_quads(clean_edges, clean_distance, foot, occlusion))
    candidates.extend(current_ellipses(clean_edges, clean_distance, foot, occlusion))
    # Foreground components give local context for textured/raised props. The
    # bbox is only a search region: its centre is never used as a landing point.
    foreground = (contrast > 25).astype(np.uint8)*255
    foreground[:int(h*.22)] = 0
    foreground[int(foot[1]+w*.03):] = 0
    components, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = sorted(components, key=cv2.contourArea, reverse=True)[:6]
    for contour in regions:
        x, y, bw, bh = cv2.boundingRect(contour)
        if not w*.10 < bw < w*.80 or bh < 30 or y > foot[1]-w*.08:
            continue
        if x <= foot[0] <= x+bw and y+bh >= foot[1]-w*.06:
            continue  # Current piece/platform already handled with an occlusion mask.
        known = [s for s in _target_surfaces(candidates, foot, w)
                 if x <= s["center"][0] <= x+bw and y <= s["center"][1] <= y+bh]
        recovered = line_quads(clean_edges, clean_distance, foot, occlusion,
                               (x-4, y-4, x+bw+4, y+bh+4), current=False)
        for candidate in recovered:
            p = np.asarray(candidate["polygon"], np.float32)
            # A closed face in this component already anchors the object.
            # Additional shadow/side-line combinations must not replace it.
            # An enclosing outer rim may replace a SMALL inset logo, however.
            if known and not any(cv2.contourArea(p) > 1.5*cv2.contourArea(np.asarray(s["polygon"], np.float32))
                                 and all(cv2.pointPolygonTest(p, tuple(map(float, v)), True) >= -3
                                         for v in s["polygon"]) for s in known):
                continue
            candidate["score"] = min(candidate["score"], .79)
            candidate["kind"] += " · 断边恢复需确认"
            candidates.append(candidate)
    return outer_surfaces(_deduplicate_surfaces(candidates))


def _surfaces(rgb: np.ndarray, foot: tuple[float, float]) -> list[dict]:
    """Compatibility helper: only feasible NEXT surfaces."""
    return _target_surfaces(_all_surfaces(rgb, foot), foot, rgb.shape[1])


def _target_surfaces(candidates: list[dict], foot, w) -> list[dict]:
    feasible = []
    for candidate in candidates:
        cx, cy = candidate["center"]
        delta_x, delta_y = abs(cx-foot[0]), foot[1]-cy
        # A foot can be off-center on its current platform. Never nominate that
        # same top face as the next destination, even if its center is above A.
        polygon = np.asarray(candidate["polygon"], dtype=np.float32)
        if cv2.pointPolygonTest(polygon, foot, True) >= -3:
            continue
        if delta_x >= w*.10 and w*.035 < delta_y < w*.65 and .20 < delta_y/delta_x < 1.18:
            feasible.append(candidate)
    return feasible


def _current_surface(surfaces: list[dict], foot, width) -> tuple[dict | None, bool]:
    owners = [s for s in surfaces if cv2.pointPolygonTest(np.asarray(s["polygon"], np.float32),
                                                         tuple(map(float, foot)), True) >= -3
              and math.dist(s["center"], foot) < width*.20]
    if not owners:
        return None, False
    owners.sort(key=lambda s: (-s["score"], math.dist(s["center"], foot)))
    best = owners[0]
    ambiguous = any(math.dist(other["center"], best["center"]) > max(8, width*.02) for other in owners[1:])
    return best, ambiguous


def _next_surface(surfaces: list[dict], foot: tuple[float, float], width: int) -> tuple[dict, bool]:
    """Only the nearest upper row may be next; quality must not skip a row.

    Competing left/right faces on that row are deliberately kept ambiguous.
    More distant upper platforms and all lower/current faces are irrelevant.
    """
    above = [s for s in surfaces if foot[1]-s["center"][1] > width*.035]
    if not above:
        raise DetectionError("当前位置上方没有可用的下一平台。")
    nearest_y = max(s["center"][1] for s in above)
    row = [s for s in above if nearest_y-s["center"][1] <= max(12, width*.04)]
    row.sort(key=lambda s: (math.dist(s["center"], foot), -s["score"]))
    return row[0], len(row) > 1


def detect(image: Image.Image) -> Detection:
    image = image.convert("RGB")
    original_w, original_h = image.size
    if original_w < 150 or original_h < 250:
        raise DetectionError("游戏区域过小；请框选完整的竖屏游戏画面。")
    if not .85 < original_h / original_w < 3.2:
        raise DetectionError("请只截取游戏区域，不要包含整张桌面。")
    scale = REFERENCE_WIDTH / original_w
    resized = image.resize((int(REFERENCE_WIDTH), round(original_h * scale)), Image.Resampling.LANCZOS)
    rgb = np.array(resized)
    pieces = _piece_candidates(rgb)
    if not pieces:
        raise DetectionError("未找到经典紫色棋子；可手动标记起点和目标，或重新截图。")
    if len(pieces) > 1 and pieces[0][0] - pieces[1][0] < .12:
        raise DetectionError("发现多个疑似棋子，已停止自动判断。请手动校正。")
    piece_conf, foot, _ = pieces[0]
    all_surfaces = _all_surfaces(rgb, foot)
    current, current_ambiguous = _current_surface(all_surfaces, foot, resized.width)
    surfaces = _target_surfaces(all_surfaces, foot, resized.width)
    if not surfaces:
        raise DetectionError("未找到完整的平台顶面；纹理、遮挡或特殊皮肤需要手动标记。")
    surface, ambiguous = _next_surface(surfaces, foot, resized.width)
    point_scale = lambda point: (float(point[0] / scale), float(point[1] / scale))
    current_quality = min(current["score"], .69 if current_ambiguous else 1.) if current else 0.
    result = Detection(piece=point_scale(foot), target=point_scale(surface["center"]),
                       confidence=min(piece_conf, surface["score"], .69 if ambiguous else 1.,
                                      current_quality if current else .65),
                       width=original_w, height=original_h,
                       polygon=tuple(point_scale(p) for p in surface["polygon"]),
                       current_center=point_scale(current["center"]) if current else None,
                       current_polygon=tuple(point_scale(p) for p in current["polygon"]) if current else (),
                       current_confidence=current_quality, target_confidence=surface["score"],
                       note=f"经典紫色棋子 · {surface['kind']} · 仅取上方最近一层 · "
                            + ("多个候选平台，需人工确认 · " if ambiguous else "")
                            + ("当前顶面未识别，需人工标 S · " if current is None else
                               "当前顶面有歧义，需人工确认 S · " if current_ambiguous else "当前顶面已定位 · ")
                            + "分数为启发式评分，不代表成功率")
    if not 45 <= result.normalized_distance <= 520:
        raise DetectionError("距离不在保守工作范围内（标准像素 45–520），请检查框选区域。")
    return result


def frame_difference(first: Image.Image, second: Image.Image) -> float:
    if first.size != second.size:
        return float("inf")
    # Ignore changing score/UI at the top and buttons at the bottom.
    w, h = first.size
    box = (0, int(h * .25), w, int(h * .85))
    a = np.array(first.convert("RGB").crop(box).resize((96, 128)), dtype=np.float32)
    b = np.array(second.convert("RGB").crop(box).resize((96, 128)), dtype=np.float32)
    return float(np.abs(a - b).mean())


def same_reviewed_frame(first: Image.Image, second: Image.Image, detection: Detection) -> bool:
    """A review authorizes one frame; verify the scene AND every marked-point ROI."""
    if first.size != second.size or first.size != (detection.width, detection.height):
        return False
    if frame_difference(first, second) > 1.0:
        return False
    radius = max(20, round(first.width * .08))
    points = (detection.piece, detection.target) + ((detection.current_center,) if detection.current_center is not None else ())
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < first.width and 0 <= y < first.height):
            return False
        box = (max(0, int(x)-radius), max(0, int(y)-radius),
               min(first.width, int(x)+radius), min(first.height, int(y)+radius))
        a = np.array(first.convert("RGB").crop(box).resize((64, 64)), dtype=np.float32)
        b = np.array(second.convert("RGB").crop(box).resize((64, 64)), dtype=np.float32)
        if float(np.abs(a-b).mean()) > 2.0:
            return False
    return True


def annotate(image: Image.Image, detection: Detection, duration: int | None = None) -> Image.Image:
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    unit = max(1.0, image.width / 540)
    width = max(2, round(3 * unit))
    if detection.polygon:
        draw.polygon(detection.polygon, outline="#fbbf24", width=width)
    if detection.current_polygon:
        draw.polygon(detection.current_polygon, outline="#4ade80", width=width)
    if detection.current_center is not None:
        draw.line([detection.current_center, detection.target], fill="#4ade80", width=max(1, width//2))
    draw.line([detection.piece, detection.target], fill="#ff537d", width=width)
    points = [(detection.target, "#fbbf24", "T")]
    if detection.current_center is not None:
        points.append((detection.current_center, "#4ade80", "S"))
    points.append((detection.piece, "#22d3ee", "A"))
    for point, color, label in points:
        x, y = point
        r = (5 if label == "A" else 10) * unit
        draw.ellipse((x - r, y - r, x + r, y + r), fill="#0c1728", outline=color, width=width)
        draw.line((x - r * 1.7, y, x + r * 1.7, y), fill=color, width=max(1, width // 2))
        draw.line((x, y - r * 1.7, x, y + r * 1.7), fill=color, width=max(1, width // 2))
        label_x = x-r*3 if label == "S" else x+r*1.8
        draw.text((label_x, y-r*2), label, fill=color, font=ImageFont.load_default(size=round(18*unit)))
    font = ImageFont.load_default(size=max(14, round(17 * unit)))
    text = f"A -> T  {detection.distance:.1f} px"
    if duration is not None:
        text += f"  |  {duration} ms"
    y = max(8, min(detection.target[1] - 60 * unit, image.height - 40 * unit))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = max(5, min((detection.piece[0] + detection.target[0]) / 2 - tw / 2, image.width - tw - 15))
    draw.rounded_rectangle((x - 8, y - 7, x + tw + 8, y + 29 * unit), radius=8 * unit, fill="#0c1728")
    draw.text((x, y), text, fill="#f1f5f9", font=font)
    if detection.center_distance is not None:
        label = f"S -> T {detection.center_distance:.1f} px"
        draw.text((12*unit, image.height-32*unit), label, fill="#4ade80", stroke_width=max(1, round(unit)),
                  stroke_fill="#0c1728", font=font)
    return result
