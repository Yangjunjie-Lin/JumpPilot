"""Top-face geometry independent of a prop's interior colour, logo or texture.

Only visible rim evidence is used. A prop's whole-body bounding-box centre is
never a landing point. All inputs use the detector's 540px reference width.
"""
from __future__ import annotations

from itertools import combinations
import math

import cv2
import numpy as np


def quad_center(vertices: np.ndarray) -> tuple[float, float]:
    """Projective centre: intersection of opposite-corner diagonals."""
    p = np.asarray(vertices, np.float64).reshape(4, 2)
    matrix = np.column_stack((p[2]-p[0], p[3]-p[1]))
    if abs(np.linalg.det(matrix)) < 1e-6:
        raise ValueError("Degenerate top face")
    t, u = np.linalg.solve(matrix, p[3]-p[0])
    if not (.05 < t < .95 and .05 < u < .95):
        raise ValueError("Non-convex top face")
    return tuple(map(float, p[0]+t*(p[2]-p[0])))


def _quad(vertices, distance, occlusion=None, source="edge") -> dict | None:
    h, w = distance.shape
    p = np.asarray(vertices, np.float32).reshape(-1, 2)
    if len(p) != 4 or not cv2.isContourConvex(p[:, None, :]):
        return None
    lo, hi = p.min(axis=0), p.max(axis=0)
    bw, bh = hi-lo
    area = cv2.contourArea(p)
    if (not w*.085 < bw < w*.66 or not 1.15 < bw/max(bh, 1) < 3.5
            or not w*w*.003 < area < w*w*.24
            or lo[0] <= 2 or hi[0] >= w-3 or lo[1] <= h*.20 or hi[1] >= h*.90):
        return None
    delta = np.roll(p, -1, axis=0)-p
    slopes = delta[:, 1]/np.where(abs(delta[:, 0]) < 1e-4, 1e-4, delta[:, 0])
    # Top faces have two rising and two falling diagonals. Vertical side faces,
    # axis-aligned labels/buttons and near-flat slivers must not qualify.
    if (np.sum(slopes > 0) != 2 or np.any(abs(slopes) < .20)
            or np.any(abs(slopes) > 1.50) or np.min(np.linalg.norm(delta, axis=1)) < 24):
        return None
    top, bottom = p[p[:, 1].argmin()], p[p[:, 1].argmax()]
    left, right = p[p[:, 0].argmin()], p[p[:, 0].argmax()]
    if abs(top[0]-bottom[0]) > bw*.40 or abs(left[1]-right[1]) > bh*.40:
        return None
    try:
        center = quad_center(p)
    except ValueError:
        return None
    supports, visible_counts = [], []
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        pts = np.rint(a+(b-a)*np.linspace(.03, .97, 48)[:, None]).astype(int)
        xx, yy = pts[:, 0], pts[:, 1]
        visible = np.ones(48, dtype=bool) if occlusion is None else ~occlusion[yy, xx]
        if visible.sum() < 20:
            return None  # Do not invent an edge hidden almost in its entirety.
        supports.append(float(np.mean(distance[yy[visible], xx[visible]] <= 2.8)))
        visible_counts.append(float(visible.mean()))
    support = float(np.mean(supports))
    if min(supports) < .74 or support < .86 or np.mean(visible_counts) < .66:
        return None
    if source == "rim" and (min(supports) < .92 or support < .97):
        return None  # Disconnected stripe/shadow lines are not four real corners.
    # Recovery is allowed to bridge only small gaps or known purple-piece
    # occlusion, and never receives more confidence than a complete contour.
    score = min(.95, .68+.27*support)
    return dict(center=center, polygon=p, score=score,
                kind="道具顶面 · 透视四角交点", source=source,
                diameter=float(bw), edge_support=support)


def contour_quads(contours, edge_distance) -> list[dict]:
    results = []
    w = edge_distance.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if not w*w*.003 < area < w*w*.24:
            continue
        hull = cv2.convexHull(contour)
        if area/max(cv2.contourArea(hull), 1) < .90:
            continue
        perimeter = cv2.arcLength(hull, True)
        for fraction in (.015, .025, .04):
            poly = cv2.approxPolyDP(hull, fraction*perimeter, True)
            if len(poly) != 4:
                continue
            surface = _quad(poly, edge_distance)
            if surface:
                results.append(surface)
                break
    return results


def piece_occlusion(rgb: np.ndarray, foot) -> np.ndarray:
    """Mask just the local purple piece, not purple graphics on other props."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (105, 35, 28), (151, 235, 175))
    yy, xx = np.ogrid[:h, :w]
    local = (abs(xx-foot[0]) < w*.065) & (yy > foot[1]-w*.24) & (yy < foot[1]+w*.035)
    mask[~local] = 0
    return cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0


def line_quads(edges, edge_distance, foot, occlusion, box=None, current=True) -> list[dict]:
    """Reconstruct four supported edges when piece/decoration opens a contour.

    Hough lines only propose geometry. All four actual rim segments must still
    pass independent pixel checks. Bounded proposals keep textured images cheap.
    """
    h, w = edges.shape
    fx, fy = foot
    if box is None:
        box = (fx-w*.34, fy-w*.26, fx+w*.34, fy+w*.25)
    x0, y0, x1, y1 = box
    x0, y0 = max(3, int(x0)), max(int(h*.22), int(y0))
    x1, y1 = min(w-3, int(x1)), min(int(h*.9), int(y1))
    if x1 <= x0 or y1 <= y0:
        return []
    roi = edges[y0:y1, x0:x1].copy()
    roi[occlusion[y0:y1, x0:x1]] = 0
    lines = cv2.HoughLinesP(roi, 1, np.pi/720, 22, minLineLength=27, maxLineGap=9)
    if lines is None:
        return []
    groups = {True: [], False: []}
    ordered = sorted(lines[:, 0], key=lambda v: math.hypot(v[2]-v[0], v[3]-v[1]), reverse=True)
    anchor_x = (x0+x1)/2
    for ax, ay, bx, by in ordered:
        if abs(int(bx)-int(ax)) < 15:
            continue
        slope = (by-ay)/(bx-ax)
        if not .20 < abs(slope) < 1.50:
            continue
        intercept = ay+y0-slope*(ax+x0)
        at_anchor = slope*anchor_x+intercept
        group = groups[slope > 0]
        if any(abs(slope-other[0]) < .10 and abs(at_anchor-other[0]*anchor_x-other[1]) < 4 for other in group):
            continue
        if len(group) < 10:
            group.append((float(slope), float(intercept)))
    def pairs(group):
        return [(a, b) for a, b in combinations(group, 2)
                if 24 < abs((a[0]-b[0])*anchor_x+a[1]-b[1]) < w*.32]
    results = []
    for positive in pairs(groups[True]):
        for negative in pairs(groups[False]):
            vertices = []
            for a, b in ((positive[0], negative[0]), (positive[0], negative[1]),
                         (positive[1], negative[1]), (positive[1], negative[0])):
                x = (b[1]-a[1])/(a[0]-b[0])
                vertices.append((x, a[0]*x+a[1]))
            p = cv2.convexHull(np.array(vertices, np.float32))[:, 0]
            if len(p) != 4 or np.any(p[:, 0] < x0-3) or np.any(p[:, 0] > x1+3) or np.any(p[:, 1] < y0-3) or np.any(p[:, 1] > y1+3):
                continue
            if current and cv2.pointPolygonTest(p, (float(fx), float(fy)), True) < -3:
                continue
            if not current:
                cx, cy = p.mean(axis=0)
                dx, dy = abs(cx-fx), fy-cy
                if dx < w*.10 or not w*.035 < dy < w*.65 or not .2 < dy/dx < 1.18:
                    continue
            surface = _quad(p, edge_distance, occlusion, "rim")
            if surface:
                results.append(surface)
    return results


def current_ellipses(edges, edge_distance, foot, occlusion) -> list[dict]:
    """Fit the visible current rim, excluding the known piece occlusion only."""
    h, w = edges.shape
    local = edges.copy()
    yy, xx = np.ogrid[:h, :w]
    local[(abs(xx-foot[0]) > w*.34) | (abs(yy-foot[1]) > w*.23) | occlusion] = 0
    contours, _ = cv2.findContours(local, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    results = []
    for contour in contours:
        if len(contour) < 70:
            continue
        points = contour[:, 0].astype(np.float32)
        try:
            (cx, cy), (a, b), deg = cv2.fitEllipse(points[:, None, :])
        except cv2.error:
            continue
        if not w*.085 < max(a, b) < w*.65 or min(a, b) < 18 or not .29 < min(a, b)/max(a, b) < .79:
            continue
        major = deg if a >= b else deg+90
        if abs((major+90) % 180-90) > 16:
            continue
        angle = math.radians(deg)
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        normalized = (points-(cx, cy)) @ rotation/(np.array([a, b])/2)
        residual = np.abs(np.linalg.norm(normalized, axis=1)-1)
        if float(np.quantile(residual, .9)) > .07:
            continue
        theta = np.linspace(0, 2*math.pi, 120, endpoint=False)
        polygon = np.column_stack((np.cos(theta)*a/2, np.sin(theta)*b/2)) @ rotation.T+(cx, cy)
        if (polygon[:, 0].min() <= 3 or polygon[:, 0].max() >= w-3
                or polygon[:, 1].min() < h*.22 or polygon[:, 1].max() >= h*.90
                or cv2.pointPolygonTest(polygon.astype(np.float32), tuple(map(float, foot)), True) < -3):
            continue
        ix, iy = np.rint(polygon).astype(int).T
        visible = ~occlusion[iy, ix]
        if visible.mean() < .66:
            continue
        support = float(np.mean(edge_distance[iy[visible], ix[visible]] <= 2.5))
        if support < .90:
            continue
        results.append(dict(center=(cx, cy), polygon=polygon, score=min(.94, .65+.29*support),
                            kind="圆形顶面 · 可见圆弧恢复", source="rim", diameter=max(a, b), edge_support=support))
    return results


def outer_surfaces(surfaces: list[dict]) -> list[dict]:
    """A logo, inset panel or concentric rim is not a separate platform.

    Containment must be complete, not just a centre test. Never suppress two
    partially overlapping actual platforms as if they were decorative nesting.
    """
    result = []
    for inner in surfaces:
        p = np.asarray(inner["polygon"], np.float32)
        area = cv2.contourArea(p)
        for outer in surfaces:
            q = np.asarray(outer["polygon"], np.float32)
            trusted_rim = outer.get("edge_support", 0) >= .97 and outer["source"] == "rim"
            if outer is inner or cv2.contourArea(q) < area*1.22 or (outer["score"] < .84 and not trusted_rim):
                continue
            if all(cv2.pointPolygonTest(q, tuple(map(float, point)), True) >= -3 for point in p):
                break
        else:
            result.append(inner)
    return result
