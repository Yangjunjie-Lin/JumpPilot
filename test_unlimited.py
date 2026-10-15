"""Unlimited/large finite runs with simulated frames and zero real input."""
from dataclasses import replace
import queue
from types import SimpleNamespace

from PIL import Image, ImageOps
import pytest

import controller
from calibration import PressModel
from controller import Runner, RunOptions
from vision import Detection, DetectionError
import wininput


@pytest.mark.parametrize("count", [0, 1, 30, 31, 100000, 10**40])
def test_nonnegative_integer_counts_have_no_upper_cap(count):
    RunOptions(PressModel(), (.5, .78), count=count).validate()


@pytest.fixture
def simulated_run(monkeypatch):
    state = SimpleNamespace(image=Image.new("RGB", (32, 48), "black"),
                            holds=[], detections=0, stop_after=100,
                            fail_detection_after=None, freeze_after=None)
    events = queue.Queue()
    runner = Runner(events)
    result = Detection((8, 36), (24, 24), .95, 32, 48)
    binding = SimpleNamespace(focus=lambda: None, verify=lambda **_: None,
                              capture=lambda: state.image)

    def check_cancel(event):
        if event.is_set():
            raise wininput.SafetyStop("模拟停止")

    def detect(image):
        state.detections += 1
        if state.fail_detection_after is not None and len(state.holds) >= state.fail_detection_after:
            raise DetectionError("模拟目标丢失")
        return result

    def hold(_binding, _point, duration, event):
        check_cancel(event)
        state.holds.append(duration)
        if state.freeze_after is None or len(state.holds) < state.freeze_after:
            state.image = ImageOps.invert(state.image)
        if len(state.holds) >= state.stop_after:
            runner.stop()
        return duration

    monkeypatch.setattr(wininput, "check_cancel", check_cancel)
    monkeypatch.setattr(wininput, "pause", lambda _seconds, event, *_: check_cancel(event))
    monkeypatch.setattr(wininput, "hold", hold)
    monkeypatch.setattr(controller, "detect", detect)
    state.runner, state.binding, state.events, state.result = runner, binding, events, result
    yield state
    runner.stop()
    if runner.thread:
        runner.thread.join(timeout=3)
        assert not runner.running


def finished_events(state, count, manual=None, reference=None):
    state.runner.start(state.binding, RunOptions(PressModel(), (.5, .78), count=count), manual, reference)
    state.runner.thread.join(timeout=3)
    assert not state.runner.running, "Simulation did not stop"
    events = list(state.events.queue)
    done = [value for kind, value in events if kind == "done"]
    assert len(done) == 1
    return events, done[0]


def test_unlimited_runs_past_30_and_ends_only_when_stopped(simulated_run):
    state = simulated_run
    state.stop_after = 75
    events, (completed, reason) = finished_events(state, 0)
    assert completed == len(state.holds) == 75
    assert reason == "模拟停止"
    assert [value["step"] for kind, value in events if kind == "jump"] == list(range(1, 76))
    progress = [value for kind, value in events if kind == "status" and "次按压" in value]
    assert len(progress) == 75 and all("/∞" in value for value in progress)
    assert state.detections == 149  # New recognition before/after every completed scene.


@pytest.mark.parametrize("count", [1, 5, 31, 80])
def test_finite_counts_finish_exactly_without_off_by_one(simulated_run, count):
    state = simulated_run
    events, (completed, reason) = finished_events(state, count)
    assert completed == len(state.holds) == count
    assert reason == "已完成计划步数。"
    assert [value["step"] for kind, value in events if kind == "jump"] == list(range(1, count+1))
    assert state.detections == 2*count  # Includes final landing-scene validation.
    assert any(kind == "status" and f"{count}/{count} 次按压" in value for kind, value in events)


def test_unlimited_stops_on_invalid_scene_after_old_limit(simulated_run):
    state = simulated_run
    state.fail_detection_after = 40
    _, (completed, reason) = finished_events(state, 0)
    assert completed == len(state.holds) == 40
    assert reason == "模拟目标丢失"


def test_unlimited_stops_when_scene_freezes_after_old_limit(simulated_run):
    state = simulated_run
    state.freeze_after = 40
    _, (completed, reason) = finished_events(state, 0)
    assert completed == len(state.holds) == 40
    assert "未明显变化" in reason


def test_unlimited_does_not_reuse_first_frame_confirmation(simulated_run):
    state = simulated_run
    state.stop_after = 40
    initial = state.result.with_confidence(.9).confirmed()
    events, _ = finished_events(state, 0, initial, state.image.copy())
    jumps = [value for kind, value in events if kind == "jump"]
    assert jumps[0]["human_confirmed"] and jumps[0]["effective_confidence"] == .9
    assert len(jumps) == 40
    assert all(not jump["human_confirmed"] and jump["effective_confidence"] == .95 for jump in jumps[1:])


def test_finite_completion_reason_not_overwritten_by_review(simulated_run, monkeypatch):
    state = simulated_run
    initial = replace(state.result, confidence=.65)
    reviews = []

    def approve(_binding, image, detection, reason, step, total):
        reviews.append((reason, step, total))
        return image, detection.confirmed()

    monkeypatch.setattr(state.runner, "_review", approve)
    _, (completed, reason) = finished_events(state, 1, initial, state.image.copy())
    assert completed == 1 and len(reviews) == 1
    assert reason == "已完成计划步数。"


def test_stopped_unlimited_runner_can_start_again(simulated_run):
    state = simulated_run
    state.stop_after = 35
    finished_events(state, 0)
    state.holds.clear()
    state.stop_after = 100
    while not state.events.empty():
        state.events.get_nowait()
    _, (completed, reason) = finished_events(state, 2)
    assert completed == len(state.holds) == 2
    assert reason == "已完成计划步数。"


def test_learning_off_never_reads_score_or_changes_model(simulated_run, monkeypatch):
    def forbidden(*_):
        pytest.fail("Learning should be opt-in")
    monkeypatch.setattr(controller, "read_score", forbidden)
    monkeypatch.setattr(controller, "measure_landing", forbidden)
    events, _ = finished_events(simulated_run, 4)
    assert len(set(simulated_run.holds)) == 1
    assert not any(kind == "learning" for kind, _ in events)


def configure_learning(state, monkeypatch):
    from adaptive import LandingFeedback
    from score_reader import ScoreReading
    monkeypatch.setattr(controller, "read_score", lambda *_: ScoreReading(len(state.holds), .95))
    monkeypatch.setattr(controller, "measure_landing", lambda *args: LandingFeedback(
        True, "模拟偏短", args[3], args[4], -20, 0, .95, (24, 24), state.result.normalized_distance))


def test_worker_applies_learned_gain_to_subsequent_jump(simulated_run, monkeypatch):
    state = simulated_run
    configure_learning(state, monkeypatch)
    state.runner.start(state.binding, RunOptions(PressModel(), (.5, .78), count=4, adaptive_learning=True))
    state.runner.thread.join(timeout=3)
    assert not state.runner.running
    assert len(state.holds) == 4
    assert len(set(state.holds[:3])) == 1 and state.holds[3] > state.holds[2]
    updates = [value for kind, value in list(state.events.queue) if kind == "learning"]
    assert [u["changed"] for u in updates] == [False, False, True, False]
    assert updates[2]["model"].k == 2.7 and updates[2]["model"].b == 0


def test_learned_gain_never_bypasses_press_duration_limit(simulated_run, monkeypatch):
    state = simulated_run
    configure_learning(state, monkeypatch)
    state.runner.start(state.binding, RunOptions(PressModel(maximum_ms=920), (.5, .78), count=0, adaptive_learning=True))
    state.runner.thread.join(timeout=3)
    assert not state.runner.running and len(state.holds) == 3
    done = [value for kind, value in list(state.events.queue) if kind == "done"][-1]
    assert done[0] == 3 and "超过安全范围" in done[1]


@pytest.mark.parametrize("mode", ["missing", "unchanged", "reset"])
def test_score_failure_skips_learning_and_reset_stops_run(simulated_run, monkeypatch, mode):
    from score_reader import ScoreReading
    state = simulated_run
    def read(*_):
        if mode == "missing":
            return ScoreReading(None)
        return ScoreReading(0 if mode == "reset" and state.holds else 10, .95)
    monkeypatch.setattr(controller, "read_score", read)
    state.runner.start(state.binding, RunOptions(PressModel(), (.5, .78), count=4, adaptive_learning=True))
    state.runner.thread.join(timeout=3)
    assert not state.runner.running
    events = list(state.events.queue)
    updates = [value for kind, value in events if kind == "learning"]
    assert updates and all(not item["changed"] for item in updates)
    assert all(not item["feedback"].accepted for item in updates)
    if mode == "reset":
        assert len(state.holds) == 1
        assert "得分下降" in [value for kind, value in events if kind == "done"][-1][1]
    else:
        assert len(state.holds) == 4
