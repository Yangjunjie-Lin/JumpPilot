"""Conservative offline OCR of a user-selected numeric game-score region.

This is a small digit-template reader, not general-purpose OCR. Unsupported
fonts, crowded UI, partial glyphs and close matches produce no usable score.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_SCORE_ROI = (.04, .12, .48, .235)


@dataclass(frozen=True)
class ScoreReading:
    value: int | None
    confidence: float = 0.
    reason: str = "未识别分数"


def validate_roi(roi):
    if len(roi) != 4 or not all(math.isfinite(v) for v in roi):
        raise ValueError("分数区域需要四个有限坐标。")
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= .30):
        raise ValueError("分数框应在画面顶部 30% 内，且左上角在右下角之前。")
    if x2-x1 < .025 or y2-y1 < .015:
        raise ValueError("分数区域过小。")


def _normalized(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    glyph = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    # Preserve aspect ratio: a narrow 1 must not become a stretched rectangle.
    scale = min(26/glyph.shape[1], 38/glyph.shape[0])
    resized = cv2.resize(glyph.astype(np.uint8)*255,
                         (max(1, round(glyph.shape[1]*scale)), max(1, round(glyph.shape[0]*scale)))) > 100
    result = np.zeros((42, 30), np.uint8)
    y, x = (42-resized.shape[0])//2, (30-resized.shape[1])//2
    result[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return result


def block_digit(digit: int) -> np.ndarray:
    """Squared seven-segment style used by the classic game score."""
    segments = ("abcedf", "bc", "abged", "abgcd", "fgbc", "afgcd", "afgecd", "abc", "abcdefg", "abfgcd")
    boxes = {"a": (4, 2, 28, 8), "b": (22, 2, 28, 25), "c": (22, 22, 28, 46),
             "d": (4, 40, 28, 46), "e": (4, 22, 10, 46), "f": (4, 2, 10, 25), "g": (4, 21, 28, 27)}
    mask = np.zeros((48, 32), np.uint8)
    for segment in segments[digit]:
        x1, y1, x2, y2 = boxes[segment]
        mask[y1:y2, x1:x2] = 1
    return mask


@lru_cache(maxsize=1)
def _templates():
    templates = [(digit, _normalized(block_digit(digit))) for digit in range(10)]
    for digit in range(10):
        for stretch in (1.2, 1.4, 1.55):
            glyph = cv2.resize(block_digit(digit), (round(32*stretch), 48), interpolation=cv2.INTER_NEAREST)
            templates.append((digit, _normalized(glyph)))
            templates.append((digit, _normalized(cv2.dilate(glyph, np.ones((3, 1), np.uint8)))))
    fonts = [ImageFont.load_default(size=48)]
    font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name in ("arial.ttf", "arialbd.ttf", "bahnschrift.ttf", "segoeui.ttf", "consolab.ttf"):
        try:
            fonts.append(ImageFont.truetype(str(font_dir/name), 48))
        except OSError:
            pass
    for font in fonts:
        for digit in range(10):
            image = Image.new("L", (90, 90))
            ImageDraw.Draw(image).text((8, 0), str(digit), font=font, fill=255)
            templates.append((digit, _normalized(np.array(image) > 110)))
    for font in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX):
        for thickness in (2, 4):
            for digit in range(10):
                image = np.zeros((80, 70), np.uint8)
                cv2.putText(image, str(digit), (5, 60), font, 1.8, 255, thickness, cv2.LINE_AA)
                templates.append((digit, _normalized(image > 110)))
    return templates


def read_score(image: Image.Image, roi=DEFAULT_SCORE_ROI) -> ScoreReading:
    validate_roi(roi)
    box = tuple(round(v*(image.width if i%2 == 0 else image.height)) for i, v in enumerate(roi))
    gray = np.array(image.convert("L").crop(box))
    if min(gray.shape) < 8 or gray.max()-int(gray.min()) < 45:
        return ScoreReading(None, reason="分数区域没有清晰数字")
    background = float(np.median(gray))
    # Scores are monochrome. Infer foreground polarity from the local background.
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = (gray < threshold+1) if background > threshold else (gray > threshold)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    components = []
    for label in range(1, n):
        x, y, w, h, area = stats[label]
        if h < max(8, gray.shape[0]*.15) or area < 12:
            continue
        if x == 0 or y == 0 or x+w >= gray.shape[1] or y+h >= gray.shape[0]:
            return ScoreReading(None, reason="分数数字被截断，请重新框选")
        if not .09 < w/h < 1.1 or h > gray.shape[0]*.95:
            return ScoreReading(None, reason="分数区域包含非数字内容")
        components.append((int(x), int(y), int(w), int(h), label))
    if not 1 <= len(components) <= 7:
        return ScoreReading(None, reason="未找到单行完整数字")
    components.sort()
    heights = [c[3] for c in components]
    bottoms = [c[1]+c[3] for c in components]
    if min(heights) < max(heights)*.72 or max(bottoms)-min(bottoms) > max(heights)*.20:
        return ScoreReading(None, reason="分数区域不是同一行数字")
    digits, confidences = [], []
    templates = _templates()
    for x, y, w, h, label in components:
        glyph = _normalized(labels[y:y+h, x:x+w] == label)
        scores = np.zeros(10)
        for digit, template in templates:
            union = np.count_nonzero(glyph | template)
            score = np.count_nonzero(glyph & template)/max(union, 1)
            scores[digit] = max(scores[digit], score)
        order = np.argsort(scores)[::-1]
        best, second = float(scores[order[0]]), float(scores[order[1]])
        if best < .78 or best-second < .065:
            return ScoreReading(None, best, "分数字形不确定，不用于学习")
        digits.append(str(order[0]))
        confidences.append(best)
    if len(digits) > 1 and digits[0] == "0":
        return ScoreReading(None, reason="疑似重复或前导零，不用于学习")
    return ScoreReading(int("".join(digits)), min(confidences), "本地数字模板识别")


def stable_score(first: ScoreReading, second: ScoreReading) -> int | None:
    return (first.value if first.value is not None and first.value == second.value
            and min(first.confidence, second.confidence) >= .85 else None)
