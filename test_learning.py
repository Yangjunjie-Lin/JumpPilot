"""Synthetic perception/learning tests; no real screen or mouse control."""
from dataclasses import replace
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytest

import adaptive
from adaptive import AdaptiveLearner, LandingFeedback, measure_landing, track_landed_platform
from calibration import PressModel
from score_reader import DEFAULT_SCORE_ROI, ScoreReading, block_digit, read_score, stable_score, validate_roi
from vision import detect, _next_surface, DetectionError


def put_score(image, number, scale=1, light=False):
    ink = (245, 245, 245) if light else (63, 64, 65)
    x, y = round(image.width*.10), round(image.height*.16)
    for character in str(number):
        glyph = Image.fromarray(block_digit(int(character))*255).resize((round(32*scale), round(48*scale)))
        image.paste(ink, (x, y), glyph)
        x += round(38*scale)


@pytest.mark.parametrize("number", list(range(10))+[10, 27, 101, 986, 5432])
@pytest.mark.parametrize("light", [False, True])
def test_numeric_score_reader(number, light):
    image = Image.new("RGB", (540, 965), (30, 30, 30) if light else (202, 204, 215))
    put_score(image, number, light=light)
    result = read_score(image)
    assert result.value == number and result.confidence >= .85


def test_supplied_screenshot_zero_score():
    image = Image.open(Path(__file__).parent/"sample.png")
    assert read_score(image).value == 0


@pytest.mark.parametrize("scale", [.5, .75, 1.5, 2])
def test_score_at_multiple_capture_sizes(scale):
    image = Image.new("RGB", (540, 965), (202, 204, 215))
    put_score(image, 278)
    image = image.resize((round(540*scale), round(965*scale)), Image.Resampling.LANCZOS)
    assert read_score(image).value == 278


def test_score_rejects_empty_cropped_and_noisy_content():
    image = Image.new("RGB", (540, 965), (202, 204, 215))
    assert read_score(image).value is None
    put_score(image, 20)
    assert read_score(image, (.10, .17, .3, .22)).value is None
    noise = np.random.default_rng(123).integers(0, 255, (965, 540, 3), dtype=np.uint8)
    assert read_score(Image.fromarray(noise)).value is None


@pytest.mark.parametrize("roi", [(0, 0, 1, 1), (.2, .2, .1, .1), (0, 0, .01, .01), (0, math.nan, .4, .2)])
def test_score_roi_validation(roi):
    with pytest.raises(ValueError):
        validate_roi(roi)


def test_score_requires_two_agreeing_high_quality_reads():
    assert stable_score(ScoreReading(10, .95), ScoreReading(10, .90)) == 10
    assert stable_score(ScoreReading(10, .95), ScoreReading(11, .90)) is None
    assert stable_score(ScoreReading(10, .80), ScoreReading(10, .90)) is None
    assert stable_score(ScoreReading(None), ScoreReading(10, .90)) is None


def candidate(x, y, score=.94):
    return {"center": (x, y), "score": score}


def test_target_selects_nearest_upper_row_not_highest_quality():
    near, far = candidate(390, 525, .80), candidate(95, 400, .99)
    selected, ambiguous = _next_surface([far, near, candidate(200, 700)], (170, 650), 540)
    assert selected is near and not ambiguous


def test_same_row_alternatives_stay_ambiguous_even_with_different_scores():
    selected, ambiguous = _next_surface([candidate(110, 500, .99), candidate(410, 505, .80)], (270, 600), 540)
    assert ambiguous and selected["center"] == (410, 505)


def test_only_lower_platforms_never_form_a_target():
    with pytest.raises(DetectionError):
        _next_surface([candidate(100, 650), candidate(400, 710)], (270, 600), 540)


def draw_platform(draw, center, style="round", radius=(70, 40)):
    x, y = center
    rx, ry = radius
    if style == "round":
        draw.ellipse((x-rx, y+35-ry, x+rx, y+35+ry), fill=(155, 156, 151))
        draw.rectangle((x-rx, y, x+rx, y+35), fill=(155, 156, 151))
        draw.ellipse((x-rx, y-ry, x+rx, y+ry), fill=(95, 101, 91))
    else:
        draw.polygon([(x-rx, y), (x, y+ry), (x+rx, y), (x+rx, y+35), (x, y+ry+35), (x-rx, y+35)], fill=(155, 156, 151))
        draw.polygon([(x-rx, y), (x, y-ry), (x+rx, y), (x, y+ry)], fill=(95, 101, 91))


def draw_piece(draw, foot):
    x, y = foot
    draw.ellipse((x-14, y-93, x+14, y-65), fill=(67, 59, 93))
    draw.polygon([(x-12, y-50), (x+12, y-50), (x+18, y-1), (x-18, y-1)], fill=(65, 54, 91))
    draw.ellipse((x-12, y-57, x+12, y-41), fill=(76, 64, 102))
    draw.ellipse((x-18, y-10, x+18, y+7), fill=(65, 54, 91))


def landing_pair(error=18, style="round", mirrored=False, camera_shift=True):
    first = Image.new("RGB", (540, 965), (202, 204, 215))
    draw = ImageDraw.Draw(first)
    draw_platform(draw, (390, 523), style)
    draw_platform(draw, (170, 650), "diamond")
    draw_piece(draw, (170, 650))
    put_score(first, 10)
    second = Image.new("RGB", first.size, (202, 204, 215))
    draw = ImageDraw.Draw(second)
    center = (320, 620) if camera_shift else (390, 523)
    draw_platform(draw, (center[0]-180, center[1]-104), "diamond")
    draw_platform(draw, center, style)
    direction = np.array((220, -127))/math.hypot(220, 127)
    draw_piece(draw, np.asarray(center)+direction*error)
    put_score(second, 11)
    if mirrored:
        first = first.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        second = second.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        center = (539-center[0], center[1])
    return first, second, center


@pytest.mark.parametrize("error", [-18, 18])
@pytest.mark.parametrize("style", ["round", "diamond"])
@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("camera_shift", [False, True])
def test_real_outline_matching_compensates_camera_translation(error, style, mirrored, camera_shift):
    first, second, expected = landing_pair(error, style, mirrored, camera_shift)
    before, after = detect(first), detect(second)
    tracked = track_landed_platform(second, before, after, first)
    assert tracked is not None
    center, quality = tracked
    assert math.dist(center, expected) < 3 and quality >= .84
    feedback = measure_landing(second, before, after, 10, 11, 700, 705, first)
    assert feedback.accepted
    assert abs(feedback.along_error-error) < 4
    assert abs(feedback.lateral_error) < 4


@pytest.mark.parametrize("scale", [.5, 1.5, 2])
def test_landing_error_is_resolution_normalized(scale):
    first, second, _ = landing_pair(-18)
    size = (round(first.width*scale), round(first.height*scale))
    first, second = (img.resize(size, Image.Resampling.LANCZOS) for img in (first, second))
    feedback = measure_landing(second, detect(first), detect(second), 10, 11, 700, 700, first)
    assert feedback.accepted and abs(feedback.along_error+18) < 4


def test_failed_platform_match_never_becomes_feedback():
    first, second, _ = landing_pair()
    before, after = detect(first), detect(second)
    blank = Image.new("RGB", second.size, (202, 204, 215))
    assert track_landed_platform(blank, before, after) is None
    assert not measure_landing(blank, before, after, 10, 11, 700, 700).accepted


@pytest.mark.parametrize("scores", [(None, 11), (10, None), (10, 10), (10, 0), (10, 999)])
def test_learning_rejects_untrusted_score_change(scores):
    first, second, _ = landing_pair()
    assert not measure_landing(second, detect(first), detect(second), *scores, 700, 700).accepted


def test_manual_override_cannot_make_learning_trust_bad_geometry():
    first, second, _ = landing_pair()
    before = replace(detect(first), confidence=.5).with_confidence(.99).confirmed()
    assert not measure_landing(second, before, detect(second), 10, 11, 700, 700).accepted
    assert not measure_landing(second, replace(detect(first), manual=True), detect(second), 10, 11, 700, 700).accepted
    assert not measure_landing(second, detect(first), detect(second), 10, 11, 700, 900).accepted


def feedback(error=18, accepted=True):
    return LandingFeedback(accepted, "test", 10, 11, error, 0, .95, (320, 620), 250)


@pytest.mark.parametrize("error", [-20, 20])
def test_learning_needs_three_consistent_samples_and_updates_in_correct_direction(error):
    baseline = PressModel(k=2.9, b=5, gain=1)
    learner = AdaptiveLearner(baseline)
    for _ in range(2):
        model, _, changed = learner.observe(feedback(error))
        assert model == baseline and not changed
    model, _, changed = learner.observe(feedback(error))
    assert changed and (model.gain-1)*error < 0
    assert abs(model.gain-1) <= .0200001
    assert model.k == baseline.k and model.b == baseline.b


def test_centered_conflicting_and_invalid_feedback_do_not_drift():
    learner = AdaptiveLearner(PressModel())
    for error in [0, 2, -2, 20, -20]*8:
        assert not learner.observe(feedback(error))[2]
    learner.observe(feedback(-20))
    learner.observe(feedback(20, accepted=False))
    assert not learner.observe(feedback(-20))[2]
    assert learner.model == PressModel()


def test_learning_is_bounded_and_memory_does_not_grow_indefinitely():
    learner = AdaptiveLearner(PressModel())
    for _ in range(300):
        learner.observe(feedback(-40))
    assert 1 <= learner.model.gain <= 1.15
    assert len(learner._batch) <= 3
    learner = AdaptiveLearner(PressModel(gain=.51))
    for _ in range(300):
        learner.observe(feedback(40))
    assert .5 <= learner.model.gain <= .51


def test_bonus_size_does_not_control_duration_change():
    first, second = AdaptiveLearner(PressModel()), AdaptiveLearner(PressModel())
    for _ in range(3):
        first.observe(feedback(-18))
        second.observe(replace(feedback(-18), score_after=30))
    assert first.model == second.model


def test_many_visible_platforms_do_not_skip_nearest_target():
    first, _, _ = landing_pair()
    draw = ImageDraw.Draw(first)
    for center in ((90, 300), (250, 350), (400, 280)):
        draw_platform(draw, center, "round", (45, 25))
    result = detect(first)
    assert math.dist(result.target, (390, 523)) < 3
    assert "最近一层" in result.note


def test_offset_foot_does_not_target_its_current_platform():
    image = Image.new("RGB", (540, 965), (202, 204, 215))
    draw = ImageDraw.Draw(image)
    draw_platform(draw, (420, 575), "round", (65, 37))
    draw_platform(draw, (170, 650), "round", (110, 63))
    draw_piece(draw, (230, 685))
    result = detect(image)
    assert math.dist(result.target, (420, 575)) < 3


def test_two_next_row_platforms_request_human_review():
    image = Image.new("RGB", (540, 965), (202, 204, 215))
    draw = ImageDraw.Draw(image)
    for center in ((120, 500), (420, 500)):
        draw_platform(draw, center, "round", (60, 34))
    draw_platform(draw, (270, 590), "diamond", (75, 43))
    draw_piece(draw, (270, 590))
    result = detect(image)
    assert result.confidence <= .69 and "多个" in result.note


def test_same_shape_different_platform_appearance_does_not_learn():
    first, second, center = landing_pair()
    draw = ImageDraw.Draw(second)
    x, y = center
    draw.ellipse((x-70, y-40, x+70, y+40), fill=(190, 60, 30))
    direction = np.array((220, -127))/math.hypot(220, 127)
    draw_piece(draw, np.asarray(center)+18*direction)
    before, after = detect(first), detect(second)
    assert track_landed_platform(second, before, after, first) is None


def test_large_lateral_deviation_never_changes_timing(monkeypatch):
    first, second, _ = landing_pair()
    before, after = detect(first), detect(second)
    unit = np.array((before.dx, before.dy))/before.distance
    center = np.asarray(after.piece)+np.array((-unit[1], unit[0]))*25
    monkeypatch.setattr(adaptive, "track_landed_platform", lambda *_: (tuple(center), .95))
    observed = measure_landing(second, before, after, 10, 11, 700, 700, first)
    assert not observed.accepted and abs(observed.lateral_error) > 20


def test_learning_rejects_nonfinite_feedback():
    learner = AdaptiveLearner(PressModel())
    for invalid in (replace(feedback(), confidence=math.nan), replace(feedback(), along_error=math.inf),
                    replace(feedback(), distance=math.nan)):
        assert not learner.observe(invalid)[2]
    assert learner.model == PressModel()


def test_offline_cli_reports_score_without_gui_or_input(tmp_path):
    import json
    import subprocess
    import sys
    directory = Path(__file__).parent
    report, output = tmp_path/"analysis.json", tmp_path/"analysis.png"
    completed = subprocess.run([sys.executable, str(directory/"app.py"), "--analyze", str(directory/"sample.png"),
                                "--report", str(report), "--output", str(output)], capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    data = json.loads(report.read_text(encoding="utf8"))
    assert data["game_score"] == 0 and data["game_score_confidence"] >= .85
    assert data["target_policy"] == "nearest_upper_row" and output.exists()


@pytest.mark.parametrize("initial_k,true_k", [(2.7, 3.0), (3.0, 2.7)])
def test_synthetic_physics_feedback_converges_into_center_deadband(initial_k, true_k):
    learner = AdaptiveLearner(PressModel(k=initial_k))
    errors, gains = [], []
    for _ in range(60):
        commanded = learner.model.predict(250, 540)
        error = commanded/true_k-250
        errors.append(error)
        previous_gain = learner.model.gain
        model, _, _ = learner.observe(replace(feedback(error), distance=250))
        assert abs(model.gain/previous_gain-1) <= .020001
        gains.append(model.gain)
    assert abs(errors[0]) > 20 and abs(errors[-1]) <= 4
    assert len(set(gains[-10:])) == 1  # Deadband prevents endless parameter drift.
