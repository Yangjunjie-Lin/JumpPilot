from dataclasses import replace
import math
from pathlib import Path
import queue
import threading

import numpy as np
from PIL import Image, ImageDraw
import pytest

from calibration import PressModel, Sample, fit_model
from controller import Runner, RunOptions
from vision import Detection, DetectionError, annotate, detect, frame_difference, load_image
import wininput

HERE = Path(__file__).resolve().parent


def sample():
    return load_image(HERE / "sample.png")


def test_reference_image():
    result = detect(sample())
    assert abs(result.piece[0]-167) < 5
    assert abs(result.piece[1]-624) < 7
    assert abs(result.target[0]-393) < 6
    assert abs(result.target[1]-492) < 6
    assert 255 < result.distance < 270
    assert result.confidence >= .84


@pytest.mark.parametrize("scale", [.5, .75, 1.25, 1.5, 2])
def test_resolution_normalization(scale):
    original = sample()
    resized = original.resize((round(original.width*scale), round(original.height*scale)), Image.Resampling.LANCZOS)
    first, second = detect(original), detect(resized)
    assert abs(first.normalized_distance-second.normalized_distance) < 5
    assert abs(PressModel().predict(first.distance, first.width)-PressModel().predict(second.distance, second.width)) < 16


def test_mirrored_target():
    image = sample().transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    result = detect(image)
    assert result.dx < 0
    assert 255 < result.distance < 270


@pytest.mark.parametrize("color", [(200, 200, 210), (0, 0, 0), (255, 255, 255)])
def test_blank_image_never_produces_input_candidate(color):
    with pytest.raises(DetectionError):
        detect(Image.new("RGB", (540, 1030), color))


def test_no_piece_rejected():
    image = sample()
    draw = ImageDraw.Draw(image)
    draw.rectangle((145, 526, 188, 635), fill=(198, 199, 209))
    with pytest.raises(DetectionError):
        detect(image)


def test_no_target_rejected():
    image = sample()
    ImageDraw.Draw(image).rectangle((280, 420, 510, 630), fill=(198, 199, 209))
    with pytest.raises(DetectionError):
        detect(image)


def test_small_and_desktop_images_rejected():
    for size in [(50, 100), (1920, 1080)]:
        with pytest.raises(DetectionError):
            detect(Image.new("RGB", size))


def test_annotation_does_not_mutate_original():
    image = sample()
    before = np.array(image)
    rendered = annotate(image, detect(image), 706)
    assert rendered.size == image.size
    assert np.array_equal(before, np.array(image))
    assert not np.array_equal(before, np.array(rendered))


def test_distance_uses_foot_and_euclidean_norm():
    detection = Detection((1, 2), (4, 6), .9, 540, 1000)
    assert detection.distance == 5
    assert detection.normalized_distance == 5
    assert detection.with_point("target", (40, 60)).manual
    with pytest.raises(ValueError):
        detection.with_point("target", (600, 50))


def test_frame_difference():
    image = sample()
    assert frame_difference(image, image.copy()) == 0
    assert frame_difference(image, Image.new("RGB", image.size)) > 2
    assert math.isinf(frame_difference(image, Image.new("RGB", (3, 4))))


def test_model_default_and_normalization():
    model = PressModel()
    assert model.predict(200, 540) == 540
    assert model.predict(400, 1080) == 540
    assert PressModel(gain=1.1).predict(200, 540) == 594


@pytest.mark.parametrize("model", [PressModel(k=-1), PressModel(k=math.nan), PressModel(b=math.inf),
                                  PressModel(gain=0), PressModel(gain=2), PressModel(maximum_ms=4000)])
def test_invalid_models_rejected(model):
    with pytest.raises(ValueError):
        model.predict(200, 540)


@pytest.mark.parametrize("distance,width", [(0, 540), (-1, 540), (math.nan, 540), (100, 0), (1000, 540), (1, 540)])
def test_invalid_and_out_of_range_duration_not_clamped(distance, width):
    with pytest.raises(ValueError):
        PressModel().predict(distance, width)


def test_fit_recovers_synthetic_linear_parameters():
    samples = [Sample(100, 540, 320), Sample(400, 1080, 620), Sample(300, 540, 920)]
    model, rmse = fit_model(samples)
    assert model.k == pytest.approx(3)
    assert model.b == pytest.approx(20)
    assert rmse < 1e-6


@pytest.mark.parametrize("samples", [[], [Sample(100, 540, 300)],
                                    [Sample(100, 540, 300), Sample(105, 540, 500)],
                                    [Sample(100, 540, 600), Sample(300, 540, 200)],
                                    [Sample(math.nan, 540, 400), Sample(300, 540, 800)]])
def test_bad_calibration_rejected(samples):
    with pytest.raises(ValueError):
        fit_model(samples)


@pytest.mark.parametrize("count", [-1, -30, True, False, 0.0, 1.5, math.nan, math.inf, "0", None])
def test_run_rejects_invalid_step_counts(count):
    with pytest.raises(ValueError):
        RunOptions(PressModel(), (.5, .78), count=count).validate()


@pytest.mark.parametrize("count", [0, 2, 31])
def test_manual_cannot_drive_multiple_jumps(count):
    runner = Runner(queue.Queue())
    with pytest.raises(ValueError):
        runner.start(None, RunOptions(PressModel(), (.5, .78), count=count),
                     replace(detect(sample()), manual=True), sample())


class FakeBinding:
    def __init__(self, image):
        self.image = image
    def focus(self):
        pass
    def verify(self, **kwargs):
        pass
    def capture(self):
        return self.image


@pytest.mark.parametrize("count", [0, 5])
def test_unchanged_frame_stops_after_one_hold(monkeypatch, count):
    holds = []
    monkeypatch.setattr(wininput, "check_cancel", lambda *_: None)
    monkeypatch.setattr(wininput, "pause", lambda *args, **kwargs: None)
    monkeypatch.setattr(wininput, "hold", lambda *args, **kwargs: holds.append(args[2]) or args[2])
    events = queue.Queue()
    runner = Runner(events)
    runner._run(FakeBinding(sample()), RunOptions(PressModel(), (.5, .78), count=count), None, None)
    assert len(holds) == 1
    done = [item for item in list(events.queue) if item[0] == "done"][-1]
    assert done[1][0] == 1
    assert "未明显变化" in done[1][1]


@pytest.mark.parametrize("count", [0, 5])
def test_invalid_detection_sends_no_input(monkeypatch, count):
    holds = []
    monkeypatch.setattr(wininput, "check_cancel", lambda *_: None)
    monkeypatch.setattr(wininput, "pause", lambda *args, **kwargs: None)
    monkeypatch.setattr(wininput, "hold", lambda *args, **kwargs: holds.append(1))
    runner = Runner(queue.Queue())
    runner._run(FakeBinding(Image.new("RGB", (540, 1030), "white")),
                RunOptions(PressModel(), (.5, .78), count=count), None, None)
    assert holds == []


def test_stale_manual_points_send_no_input(monkeypatch):
    holds = []
    monkeypatch.setattr(wininput, "check_cancel", lambda *_: None)
    monkeypatch.setattr(wininput, "pause", lambda *args, **kwargs: None)
    monkeypatch.setattr(wininput, "hold", lambda *args, **kwargs: holds.append(1))
    runner = Runner(queue.Queue())
    runner._run(FakeBinding(Image.new("RGB", sample().size, "white")),
                RunOptions(PressModel(), (.5, .78), count=1),
                replace(detect(sample()), manual=True), sample())
    assert holds == []


def test_hold_always_releases_on_cancellation(monkeypatch):
    if not wininput.IS_WINDOWS:
        pytest.skip("Windows structure test")
    calls = []
    class FakeUser:
        def GetAsyncKeyState(self, _): return 0
        def GetCursorPos(self, pointer):
            pointer._obj.x = 100
            pointer._obj.y = 100
            return 1
        def SetCursorPos(self, *args): return 1
    class FakeInputBinding:
        hwnd = 123
        def verify(self, **kwargs): pass
        def screen_point(self, _): return 100, 100
    def cancel(*args):
        raise wininput.SafetyStop("test cancel")
    monkeypatch.setattr(wininput, "user32", FakeUser())
    monkeypatch.setattr(wininput, "check_cancel", lambda *_: None)
    monkeypatch.setattr(wininput, "root_at", lambda *args: 123)
    monkeypatch.setattr(wininput, "_mouse", calls.append)
    monkeypatch.setattr(wininput, "pause", cancel)
    with pytest.raises(wininput.SafetyStop):
        wininput.hold(FakeInputBinding(), (.5, .78), 400, threading.Event())
    assert calls == [0x0002, 0x0004]


def test_hold_rejects_unsafe_duration_before_input():
    with pytest.raises(wininput.SafetyStop):
        wininput.hold(None, (.5, .78), 99999, threading.Event())


def test_runner_thread_stop_is_interruptible(monkeypatch):
    holds = []
    def pause(_seconds, event, *_args):
        event.wait(.01)
        if event.is_set():
            raise wininput.SafetyStop("stop requested")
    monkeypatch.setattr(wininput, "pause", pause)
    monkeypatch.setattr(wininput, "hold", lambda *args: holds.append(1))
    runner = Runner(queue.Queue())
    runner.start(FakeBinding(sample()), RunOptions(PressModel(), (.5, .78), count=5))
    runner.stop()
    runner.thread.join(timeout=2)
    assert not runner.running
    assert not holds


def test_binding_rejects_window_move(monkeypatch):
    if not wininput.IS_WINDOWS:
        pytest.skip("Windows-only binding")
    class FakeUser:
        def IsWindow(self, _): return True
        def IsWindowVisible(self, _): return True
        def IsIconic(self, _): return False
    monkeypatch.setattr(wininput, "user32", FakeUser())
    monkeypatch.setattr(wininput, "process_id", lambda _: 5)
    monkeypatch.setattr(wininput, "window_rect", lambda _: (1, 0, 601, 1000))
    binding = wininput.Binding(123, 5, "test", (0, 0, 600, 1000), (20, 30, 560, 980))
    with pytest.raises(wininput.SafetyStop, match="位置或大小"):
        binding.verify()


def test_binding_rejects_lost_focus(monkeypatch):
    if not wininput.IS_WINDOWS:
        pytest.skip("Windows-only binding")
    class FakeUser:
        def IsWindow(self, _): return True
        def IsWindowVisible(self, _): return True
        def IsIconic(self, _): return False
        def GetForegroundWindow(self): return 999
    monkeypatch.setattr(wininput, "user32", FakeUser())
    monkeypatch.setattr(wininput, "process_id", lambda _: 5)
    monkeypatch.setattr(wininput, "window_rect", lambda _: (0, 0, 600, 1000))
    binding = wininput.Binding(123, 5, "test", (0, 0, 600, 1000), (20, 30, 560, 980))
    with pytest.raises(wininput.SafetyStop, match="焦点"):
        binding.verify()


def test_binding_rejects_occlusion(monkeypatch):
    if not wininput.IS_WINDOWS:
        pytest.skip("Windows-only binding")
    class FakeUser:
        def IsWindow(self, _): return True
        def IsWindowVisible(self, _): return True
        def IsIconic(self, _): return False
        def GetForegroundWindow(self): return 123
    monkeypatch.setattr(wininput, "user32", FakeUser())
    monkeypatch.setattr(wininput, "process_id", lambda _: 5)
    monkeypatch.setattr(wininput, "window_rect", lambda _: (0, 0, 600, 1000))
    monkeypatch.setattr(wininput, "root_at", lambda *_: 999)
    binding = wininput.Binding(123, 5, "test", (0, 0, 600, 1000), (20, 30, 560, 980))
    with pytest.raises(wininput.SafetyStop, match="遮挡"):
        binding.verify()


def test_gui_offline_never_arms_or_runs(monkeypatch, tmp_path, gui_root):
    import tkinter as tk
    import app as gui
    monkeypatch.setattr(gui, "APP_DIR", tmp_path)
    root = tk.Toplevel(gui_root)
    root.withdraw()
    try:
        application = gui.App(root)
        assert application.source == "file"
        assert application.binding is None
        assert not application.armed.get()
        assert application.result is not None
        assert application.time_text.get() == "707"  # v1.4 uses the quadrilateral's diagonal intersection.
        with pytest.raises(ValueError, match="离线图片"):
            application.start(1)
        assert not application.runner.running
        application.armed.set(True)
        application.k.set("3.0")
        assert not application.armed.get()
        application.save_config()
        import json
        data = json.loads((tmp_path / "jump-pilot-settings.json").read_text(encoding="utf8"))
        assert data["model"]["k"] == 3.0
        assert "binding" not in data and "armed" not in data
    finally:
        root.destroy()
