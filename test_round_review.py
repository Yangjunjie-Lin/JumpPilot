"""Round-platform geometry and one-frame human-review regression tests.

All input functions are mocked. Synthetic ground truth is not a claim of
real-game success rate; the supplied screenshot already has overlays on it.
"""
from dataclasses import replace
import math
from pathlib import Path
import queue
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytest

import controller
from controller import Runner, RunOptions, ReviewRequest, needs_review
from calibration import PressModel
from vision import Detection, DetectionError, detect, _ellipse_surface, same_reviewed_frame, frame_difference
import wininput


def round_scene(style="ring", center=(166, 430), radii=(67, 38), mirrored=False):
    image = Image.new("RGB", (540, 965), (202, 204, 215))
    draw = ImageDraw.Draw(image)
    draw.polygon([(295,563),(390,508),(485,563),(390,618)], fill=(243,243,243))
    draw.polygon([(295,563),(390,618),(390,654),(295,599)], fill=(156,156,156))
    draw.polygon([(485,563),(390,618),(390,654),(485,599)], fill=(181,181,181))
    cx, cy = center
    rx, ry = radii
    draw.ellipse((cx-rx-58,cy-3,cx+rx-15,cy+77), fill=(141,144,151))
    draw.ellipse((cx-rx,cy+62-ry,cx+rx,cy+62+ry), fill=(184,184,182))
    draw.rectangle((cx-rx,cy,cx+rx,cy+62), fill=(184,184,182))
    draw.ellipse((cx-rx,cy-ry,cx+rx,cy+ry), fill=(112,113,111))
    if style == "ring":
        draw.ellipse((cx-rx+5,cy-ry+4,cx+rx-5,cy+ry-4), fill=(186,181,147))
        draw.ellipse((cx-rx+9,cy-ry+8,cx+rx-9,cy+ry-8), fill=(106,107,105))
        draw.ellipse((cx-31,cy-17,cx+31,cy+17), fill=(155,155,150))
    if style == "textured":
        for dx in range(-40,50,15):
            draw.line((cx+dx,cy-18,cx+dx+4,cy+18), fill=(90,93,90), width=4)
    if style == "gradient":
        arr = np.array(image)
        yy, xx = np.mgrid[:965,:540]
        mask = ((xx-cx)/rx)**2+((yy-cy)/ry)**2 <= .98
        values = (100+30*(xx-cx)/rx).astype(np.uint8)
        for channel in range(3):
            arr[:,:,channel][mask] = values[mask]
        image = Image.fromarray(arr)
        draw = ImageDraw.Draw(image)
    draw.ellipse((376,470,404,498), fill=(67,59,93))
    draw.polygon([(378,513),(402,513),(408,562),(372,562)], fill=(65,54,91))
    draw.ellipse((378,506,402,522), fill=(76,64,102))
    draw.ellipse((372,553,408,570), fill=(65,54,91))
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if mirrored else image


@pytest.mark.parametrize("style", ["plain", "ring", "textured", "gradient"])
@pytest.mark.parametrize("scale", [.5, 1, 1.5, 2])
@pytest.mark.parametrize("mirrored", [False, True])
def test_round_centers_across_themes_scales_and_directions(style, scale, mirrored):
    original = round_scene(style, mirrored=mirrored)
    image = original.resize((round(540*scale), round(965*scale)), Image.Resampling.LANCZOS)
    result = detect(image)
    expected_x = 539-166 if mirrored else 166
    error = math.hypot(result.target[0]*540/image.width-expected_x,
                       result.target[1]*540/image.width-430)
    assert error < 1.5
    assert result.confidence >= .84
    assert "圆形" in result.note
    assert result.confidence_override is None
    assert not result.human_confirmed


@pytest.mark.parametrize("radii", [(43, 23), (53, 30), (83, 43), (100, 58)])
def test_round_sizes_and_concentric_rims_are_not_ambiguous(radii):
    result = detect(round_scene(radii=radii))
    assert np.linalg.norm(np.subtract(result.target, (166,430))) < 2
    assert result.confidence > .84
    assert "多个" not in result.note


def test_round_rejects_clipped_platform():
    with pytest.raises(DetectionError):
        detect(round_scene(center=(12,430), style="plain"))


def test_round_with_mild_pixel_noise():
    image = round_scene("gradient")
    noise = np.random.default_rng(42).normal(0, 1.2, (965,540,3))
    result = detect(Image.fromarray(np.clip(np.array(image)+noise, 0, 255).astype(np.uint8)))
    assert np.linalg.norm(np.subtract(result.target, (166,430))) < 2


def test_incomplete_contour_does_not_get_a_good_ellipse_score():
    theta = np.linspace(0, math.pi*1.4, 90)
    contour = np.rint(np.column_stack((166+67*np.cos(theta), 430+38*np.sin(theta)))).astype(np.int32)[:,None,:]
    assert _ellipse_surface(contour, np.zeros((965,540), dtype=np.float32), "edge") is None


def test_rectangle_not_mistaken_for_round_platform():
    contour = np.array([(99,392),(233,392),(233,468),(99,468)], dtype=np.int32)[:,None,:]
    assert _ellipse_surface(contour, np.zeros((965,540), dtype=np.float32), "edge") is None


def test_override_preserves_raw_score_and_is_cleared_on_point_change():
    original = detect(round_scene())
    revised = original.with_confidence(.60).confirmed()
    assert revised.confidence == original.confidence
    assert revised.effective_confidence == .60
    assert revised.human_confirmed
    changed = revised.with_point("target", (166,431))
    assert changed.confidence == original.confidence
    assert changed.confidence_override is None and not changed.human_confirmed
    assert changed.manual and needs_review(changed, .1)


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.01, 1.01])
def test_invalid_confidence_override_is_rejected(value):
    with pytest.raises(ValueError):
        detect(round_scene()).with_confidence(value)


@pytest.mark.parametrize("threshold", [0, .60, .84, 1])
def test_user_threshold_range(threshold):
    RunOptions(PressModel(), (.5,.78), minimum_confidence=threshold).validate()


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -0.1, 1.1])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        RunOptions(PressModel(), (.5,.78), minimum_confidence=threshold).validate()


def test_confirmation_overrides_gate_once_not_score():
    detection = replace(detect(round_scene()), confidence=.52)
    assert needs_review(detection, .84)
    confirmed = detection.confirmed()
    assert confirmed.effective_confidence == .52
    assert not needs_review(confirmed, .84)
    assert needs_review(detection.with_confidence(.70), .84)
    assert not needs_review(detection.with_confidence(.90), .84)


def test_local_target_change_invalidates_review_even_with_small_global_change():
    first = round_scene()
    changed = first.copy()
    ImageDraw.Draw(changed).rectangle((159,423,173,437), fill="white")
    assert frame_difference(first, changed) < 1.0
    assert not same_reviewed_frame(first, changed, detect(first))
    assert same_reviewed_frame(first, first.copy(), detect(first))


class Binding:
    def __init__(self):
        self.image = round_scene()
        self.focus_count = 0
    def capture(self): return self.image
    def verify(self, **kwargs): pass
    def focus(self): self.focus_count += 1


def setup_runner(monkeypatch, count=2, threshold=.84):
    binding = Binding()
    events = queue.Queue()
    runner = Runner(events)
    holds = []
    real_detect = detect
    monkeypatch.setattr(controller, "detect", lambda image: replace(real_detect(image), confidence=.65))
    def check_cancel(event):
        if event.is_set(): raise wininput.SafetyStop("test stopped")
    monkeypatch.setattr(wininput, "check_cancel", check_cancel)
    monkeypatch.setattr(wininput, "pause", lambda _seconds, event, *_args: check_cancel(event))
    def hold(_binding, _point, duration, _stop):
        holds.append(duration)
        binding.image = binding.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return duration
    monkeypatch.setattr(wininput, "hold", hold)
    runner.start(binding, RunOptions(PressModel(), (.5,.78), count=count, minimum_confidence=threshold))
    return runner, binding, events, holds


def next_event(events, kind, timeout=3):
    import time
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        event = events.get(timeout=max(.01,deadline-time.monotonic()))
        if event[0] == kind: return event[1]
    raise AssertionError(f"No {kind} event")


@pytest.mark.parametrize("count", [0, 2])
def test_low_confidence_waits_without_input_then_reviews_next_frame(monkeypatch, count):
    runner, binding, events, holds = setup_runner(monkeypatch, count=count)
    try:
        first = next_event(events, "review")
        assert first.step == 1 and first.total == count
        assert not holds
        assert runner.pending_review.request_id == first.request_id
        with pytest.raises(ValueError, match="过期"):
            runner.approve(first.request_id+100, first.detection)
        runner.approve(first.request_id, first.detection.with_confidence(.92))
        jump = next_event(events, "jump")
        assert jump["raw_confidence"] == .65 and jump["effective_confidence"] == .92
        assert jump["human_confirmed"] and len(holds) == 1
        second = next_event(events, "review")
        assert second.step == 2 and second.total == count
        assert second.request_id != first.request_id
        assert second.detection.confidence_override is None
        assert not second.detection.human_confirmed
        assert len(holds) == 1
        assert binding.focus_count == 2  # Start and explicit review resume.
    finally:
        runner.stop()
        runner.thread.join(timeout=2)
    assert not runner.running and runner.pending_review is None


@pytest.mark.parametrize("count", [0, 2])
def test_stop_while_waiting_never_sends_input(monkeypatch, count):
    runner, _, events, holds = setup_runner(monkeypatch, count=count)
    request = next_event(events, "review")
    runner.stop()
    runner.thread.join(timeout=2)
    assert not runner.running and not holds
    with pytest.raises(ValueError, match="过期"):
        runner.approve(request.request_id, request.detection)


@pytest.mark.parametrize("count", [0, 2])
def test_changed_frame_after_confirmation_sends_no_input(monkeypatch, count):
    runner, binding, events, holds = setup_runner(monkeypatch, count=count)
    request = next_event(events, "review")
    binding.image = binding.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    runner.approve(request.request_id, request.detection)
    completed, reason = next_event(events, "done")
    runner.thread.join(timeout=2)
    assert completed == 0 and not holds
    assert "确认已作废" in reason


def test_lower_user_threshold_allows_original_score_without_override(monkeypatch):
    runner, _, events, holds = setup_runner(monkeypatch, count=1, threshold=.6)
    completed, _ = next_event(events, "done")
    runner.thread.join(timeout=2)
    assert completed == 1 and len(holds) == 1
    assert runner.pending_review is None


@pytest.fixture
def gui_app(monkeypatch, tmp_path, gui_root):
    import tkinter as tk
    import app as gui
    monkeypatch.setattr(gui, "APP_DIR", tmp_path)
    root = tk.Toplevel(gui_root)
    root.withdraw()
    app = gui.App(root)
    yield app, tmp_path
    app.runner.stop()
    if app.runner.thread: app.runner.thread.join(timeout=2)
    root.destroy()


def test_gui_override_and_confirmation_do_not_arm_or_click(gui_app):
    app, _ = gui_app
    original = app.result.confidence
    app.manual_score.set("75")
    app.apply_score()
    assert app.result.confidence == original
    assert app.result.effective_confidence == .75
    assert "人工" in app.detail.get()
    assert not app.armed.get()
    app.confirm_frame()
    assert app.result.human_confirmed
    assert not app.runner.running
    with pytest.raises(ValueError, match="离线"):
        app.start(1)
    app.show_frame(round_scene())
    assert app.result.confidence_override is None and not app.result.human_confirmed
    assert not app.manual_score.get()


def test_gui_persists_threshold_but_never_per_frame_review(gui_app):
    import json
    app, path = gui_app
    app.confidence_threshold.set("72")
    app.manual_score.set("90")
    app.apply_score()
    app.confirm_frame()
    app.save_config()
    data = json.loads((path / "jump-pilot-settings.json").read_text(encoding="utf8"))
    assert data["minimum_confidence"] == .72
    assert not any(key in data for key in ("confidence_override", "human_confirmed", "review_request"))


def test_gui_review_enables_only_review_controls(gui_app, monkeypatch):
    app, _ = gui_app
    result = replace(app.result, confidence=.62)
    app.show_frame(app.image, result, analyse=False)
    app.review_request = ReviewRequest(123, app.image.copy(), result, "test", 1, 5)
    app._set_busy(True)
    for control in app.review_controls:
        assert not control.instate(["disabled"])
    locked = [c for c in app.controls if c not in app.review_controls]
    assert locked and all(c.instate(["disabled"]) for c in locked)
    approvals = []
    monkeypatch.setattr(app.runner, "approve", lambda request_id, detection: approvals.append((request_id,detection)))
    app.manual_score.set("68")
    app.confirm_frame()
    assert approvals[0][0] == 123
    assert approvals[0][1].effective_confidence == .68
    assert approvals[0][1].confidence == .62
    assert approvals[0][1].human_confirmed
    assert app.review_request is None
    assert all(c.instate(["disabled"]) for c in app.review_controls)


@pytest.mark.parametrize("entered,argument,expected", [("0", None, 0), ("1000", None, 1000), ("0", 1, 1)])
def test_gui_passes_unlimited_and_finite_counts_without_input(gui_app, monkeypatch, entered, argument, expected):
    from types import SimpleNamespace
    app, _ = gui_app
    app.binding = SimpleNamespace(verify=lambda **_: None, screen_point=lambda _: (100, 100))
    app.source = "screen"
    app.armed.set(True)
    app.count.set(entered)
    starts = []
    monkeypatch.setattr(app.runner, "start", lambda *args: starts.append(args))
    app.start(argument)
    assert len(starts) == 1 and starts[0][1].count == expected
    assert app.busy and not app.runner.running


@pytest.mark.parametrize("entered", ["-1", "1.5", "无限", "", "NaN", "inf"])
def test_gui_rejects_invalid_step_counts_before_start(gui_app, monkeypatch, entered):
    from types import SimpleNamespace
    app, _ = gui_app
    app.binding = SimpleNamespace(verify=lambda **_: None, screen_point=lambda _: (100, 100))
    app.source = "screen"
    app.armed.set(True)
    app.count.set(entered)
    starts = []
    monkeypatch.setattr(app.runner, "start", lambda *args: starts.append(args))
    with pytest.raises(ValueError, match="非负整数"):
        app.start(None)
    assert not starts and not app.busy


def test_gui_unlimited_review_uses_infinity_not_zero(gui_app, monkeypatch):
    app, _ = gui_app
    # Keep this regression test hidden; never activate or manipulate desktop windows.
    for action in ("deiconify", "lift", "focus_force"):
        monkeypatch.setattr(app.root, action, lambda: None)
    request = ReviewRequest(123, app.image.copy(), app.result, "低置信度", 31, 0)
    app.events.put(("review", request))
    app._poll()
    assert "31/∞" in app.review_text.get()
    assert "31/0" not in app.review_text.get()
    assert app.review_request is request and app.busy


def test_gui_learning_opt_in_options_and_no_automatic_persistence(gui_app, monkeypatch):
    from types import SimpleNamespace
    import json
    app, path = gui_app
    assert not app.adaptive_enabled.get()
    app.binding = SimpleNamespace(verify=lambda **_: None, screen_point=lambda _: (100, 100))
    app.source = "screen"
    app.adaptive_enabled.set(True)
    app.armed.set(True)
    starts = []
    monkeypatch.setattr(app.runner, "start", lambda *args: starts.append(args))
    app.start(1)
    assert starts[0][1].adaptive_learning
    assert starts[0][1].score_roi == app.score_roi
    assert app.learning_baseline == app.model()
    assert not (path/"jump-pilot-settings.json").exists()
    app._set_busy(False)
    app.save_config()
    data = json.loads((path/"jump-pilot-settings.json").read_text(encoding="utf8"))
    assert "score_roi" in data
    assert not any(key in data for key in ("adaptive_learning", "learning_history", "armed"))


def test_gui_applies_learning_event_and_undo_restores_baseline(gui_app):
    app, path = gui_app
    baseline = app.model()
    app.learning_baseline = baseline
    changed = replace(baseline, gain=baseline.gain*1.02)
    app.events.put(("learning", {"changed": True, "model": changed, "message": "模拟偏短",
                                 "step": 3, "accepted": 3, "updates": 1}))
    app._poll()
    assert app.model().gain == pytest.approx(changed.gain)
    assert app.model().k == baseline.k and app.model().b == baseline.b
    assert "模拟偏短" in app.learning_text.get()
    assert not (path/"jump-pilot-settings.json").exists()
    app.undo_learning()
    assert app.model() == baseline and not app.adaptive_enabled.get()
    assert not app.armed.get()


def test_score_region_selection_does_not_change_jump_points(gui_app):
    from types import SimpleNamespace
    app, _ = gui_app
    original = app.result
    app.begin_score_region()
    app.image_transform = (1, 0, 0)
    app.pick_point(SimpleNamespace(x=40, y=150))
    app.pick_point(SimpleNamespace(x=240, y=235))
    assert app.mode.get() == "inspect" and app.score_corner is None
    assert app.result is original and not app.result.manual
    assert "游戏得分：0" in app.game_score_text.get()


def test_gui_learning_history_is_bounded(gui_app):
    app, _ = gui_app
    for step in range(120):
        app.events.put(("learning", {"changed": False, "message": "不学习", "model": app.model(),
                                     "step": step, "accepted": 0, "updates": 0}))
    app._poll()
    assert len(app.learning_history) == 100
    assert app.learning_history[0]["step"] == 20


def test_undo_learning_does_not_overwrite_manual_k_and_b(gui_app):
    app, _ = gui_app
    app.learning_baseline = app.model()
    app.gain.set("1.02")
    app.k.set("3.1")
    app.b.set("15")
    app.undo_learning()
    assert app.model().k == 3.1 and app.model().b == 15
    assert app.model().gain == app.learning_baseline.gain
