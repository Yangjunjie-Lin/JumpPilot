"""Single worker automation; no GUI calls from its background thread."""
from __future__ import annotations

from dataclasses import dataclass
import math
import queue
import threading
import time

from PIL import Image

from calibration import PressModel
from adaptive import AdaptiveLearner, LandingFeedback, measure_landing
from score_reader import DEFAULT_SCORE_ROI, read_score, stable_score, validate_roi
from vision import Detection, detect, frame_difference, same_reviewed_frame
import wininput


@dataclass(frozen=True)
class RunOptions:
    model: PressModel
    relative_click: tuple[float, float]
    count: int = 1  # 0 means no step-count limit; safety/cancellation checks still apply.
    settle_seconds: float = 1.5
    minimum_confidence: float = .84
    adaptive_learning: bool = False
    score_roi: tuple[float, float, float, float] = DEFAULT_SCORE_ROI

    def validate(self) -> None:
        self.model.validate()
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise ValueError("本轮步数必须是非负整数；0 表示无限跳，正整数表示指定次数。")
        if not 1 <= self.settle_seconds <= 5:
            raise ValueError("每跳等待必须在 1–5 秒之间。")
        if not math.isfinite(self.minimum_confidence) or not 0 <= self.minimum_confidence <= 1:
            raise ValueError("自动放行阈值必须在 0%–100% 之间。")
        if not isinstance(self.adaptive_learning, bool):
            raise ValueError("自适应学习开关必须是布尔值。")
        validate_roi(self.score_roi)


@dataclass(frozen=True)
class ReviewRequest:
    request_id: int
    image: Image.Image
    detection: Detection
    reason: str
    step: int
    total: int


def needs_review(detection: Detection, threshold: float) -> bool:
    return not detection.human_confirmed and (detection.manual or detection.effective_confidence < threshold)


class Runner:
    def __init__(self, events: queue.Queue):
        self.events = events
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._review_lock = threading.Lock()
        self._review_event = threading.Event()
        self._pending: ReviewRequest | None = None
        self._response: Detection | None = None
        self._request_counter = 0

    @property
    def pending_review(self) -> ReviewRequest | None:
        with self._review_lock:
            return self._pending

    def approve(self, request_id: int, detection: Detection) -> None:
        """A response is accepted once, only for the currently displayed frame."""
        with self._review_lock:
            request = self._pending
            if self.stop_event.is_set() or request is None or request.request_id != request_id:
                raise ValueError("人工确认已过期，请重新识别。")
            if self._review_event.is_set():
                raise ValueError("本帧已确认，请等待倒计时。")
            if (detection.width, detection.height) != request.image.size:
                raise ValueError("确认的画面尺寸不一致。")
            points = (detection.piece, detection.target) + ((detection.current_center,) if detection.current_center is not None else ())
            for point in points:
                if not all(math.isfinite(value) for value in point):
                    raise ValueError("标记点无效。")
                if not (0 <= point[0] < detection.width and 0 <= point[1] < detection.height):
                    raise ValueError("标记点不在画面内。")
            detection.with_confidence(detection.confidence_override)  # Validate override.
            self._response = detection.confirmed()
            self._review_event.set()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, binding: wininput.Binding, options: RunOptions,
              manual: Detection | None = None, reference: Image.Image | None = None) -> None:
        if self.running:
            raise ValueError("已有任务正在运行。")
        options.validate()
        if manual is not None and (reference is None or (manual.manual and options.count != 1)):
            raise ValueError("手动标记只能对当前屏幕截图进行单步试跳。")
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, args=(binding, options, manual, reference),
                                       daemon=True, name="JumpPilot-input")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _countdown(self, binding: wininput.Binding, *, resumed: bool = False) -> None:
        binding.focus()
        for remaining in (3, 2, 1):
            prefix = "已确认本帧，" if resumed else ""
            self.events.put(("status", f"{prefix}{remaining} 秒后开始；Esc 随时停止"))
            wininput.pause(1, self.stop_event)
        binding.verify()

    def _review(self, binding: wininput.Binding, image: Image.Image, detection: Detection,
                reason: str, step: int, total: int) -> tuple[Image.Image, Detection]:
        with self._review_lock:
            self._request_counter += 1
            request = ReviewRequest(self._request_counter, image.copy(), detection, reason, step, total)
            self._review_event.clear()
            self._response = None
            self._pending = request
        self.events.put(("review", request))
        try:
            # No input is held; the assistant UI may receive focus while reviewing.
            # Stop/Esc remains live, but focus/occlusion checks resume only AFTER
            # the user's confirmation, so clicking Confirm cannot self-cancel.
            while not self._review_event.wait(.05):
                wininput.check_cancel(self.stop_event)
            wininput.check_cancel(self.stop_event)
            with self._review_lock:
                approved = self._response
            if approved is None:
                raise wininput.SafetyStop("未收到有效的人工确认。")
        finally:
            with self._review_lock:
                self._pending = None
                self._response = None
            self.events.put(("review_closed", request.request_id))
        self._countdown(binding, resumed=True)
        current = self._stable_frame(binding)
        if not same_reviewed_frame(image, current, approved):
            raise wininput.SafetyStop("确认后游戏画面发生变化，本次确认已作废；请重新识别。")
        return current, approved

    def _stable_frame(self, binding: wininput.Binding) -> Image.Image:
        binding.verify()
        previous = binding.capture()
        for _ in range(8):
            wininput.pause(.22, self.stop_event, binding)
            frame = binding.capture()
            if frame_difference(previous, frame) < 1.5:
                return frame
            previous = frame
        raise wininput.SafetyStop("画面仍在运动或切换，未执行按压。")

    def _run(self, binding: wininput.Binding, options: RunOptions,
             manual: Detection | None, reference: Image.Image | None) -> None:
        completed = 0
        reason = "已完成计划步数。"
        total_label = "∞" if options.count == 0 else str(options.count)
        learner = AdaptiveLearner(options.model)
        try:
            self._countdown(binding)
            while options.count == 0 or completed < options.count:
                wininput.check_cancel(self.stop_event)
                frame = self._stable_frame(binding)
                if manual is not None and completed == 0:
                    if not same_reviewed_frame(reference, frame, manual):
                        raise wininput.SafetyStop("屏幕已变化，人工标记/评分/确认失效；请重新识别。")
                    detection = manual
                else:
                    detection = detect(frame)
                if needs_review(detection, options.minimum_confidence):
                    review_reason = ("手动修正了坐标，请核对本次落点。" if detection.manual else
                                     f"有效置信度 {detection.effective_confidence:.0%} 低于自动阈值 {options.minimum_confidence:.0%}。")
                    frame, detection = self._review(binding, frame, detection, review_reason,
                                                    completed+1, options.count)
                duration = learner.model.predict(detection.distance, frame.width)
                self.events.put(("preview", (frame, detection, duration)))
                # Reject changes since recognition rather than clicking from a stale image.
                wininput.pause(.08, self.stop_event, binding)
                prepress = binding.capture()
                if not same_reviewed_frame(frame, prepress, detection):
                    raise wininput.SafetyStop("识别后画面变化，取消本次按压。")
                score_before = (stable_score(read_score(frame, options.score_roi), read_score(prepress, options.score_roi))
                                if options.adaptive_learning else None)
                actual = wininput.hold(binding, options.relative_click, duration, self.stop_event)
                completed += 1
                self.events.put(("jump", {"step": completed, "distance": detection.distance,
                                         "width": frame.width, "duration": duration,
                                         "actual_ms": actual, "at": time.time(),
                                         "raw_confidence": detection.confidence,
                                         "effective_confidence": detection.effective_confidence,
                                         "human_confirmed": detection.human_confirmed}))
                self.events.put(("status", f"已执行 {completed}/{total_label} 次按压；等待画面稳定"))
                wininput.pause(options.settle_seconds, self.stop_event, binding)
                after = self._stable_frame(binding)
                if frame_difference(frame, after) < 2.0:
                    raise wininput.SafetyStop("按压后画面未明显变化，已停止以避免重复输入。")
                # Even the last jump gets checked. This is visual scene validation,
                # NOT proof that the game awarded a successful landing.
                next_detection = detect(after)
                if options.adaptive_learning:
                    wininput.pause(.18, self.stop_event, binding)
                    verified = binding.capture()
                    score_after = stable_score(read_score(after, options.score_roi), read_score(verified, options.score_roi))
                    if same_reviewed_frame(after, verified, next_detection):
                        feedback = measure_landing(verified, detection, next_detection,
                                                   score_before, score_after, duration, actual, frame)
                    else:
                        feedback = LandingFeedback(False, "落地后画面仍在变化，本跳不学习", score_before, score_after)
                    model, message, changed = learner.observe(feedback)
                    self.events.put(("learning", {"feedback": feedback, "model": model, "message": message,
                                                  "changed": changed, "accepted": learner.accepted,
                                                  "updates": learner.updates, "step": completed}))
                    if score_before is not None and score_after is not None and score_after < score_before:
                        raise wininput.SafetyStop("游戏得分下降或重置，已停止；请检查是否结束或进入新一局。")
                self.events.put(("preview", (after, next_detection, None)))
                if options.count > 0 and completed == options.count and needs_review(next_detection, options.minimum_confidence):
                    reason = "计划步数已结束；下一帧评分较低，下次运行时将请求人工确认。"
        except Exception as error:
            reason = str(error) or type(error).__name__
        finally:
            self.stop_event.set()
            with self._review_lock:
                self._pending = None
                self._response = None
            self.events.put(("done", (completed, reason)))
