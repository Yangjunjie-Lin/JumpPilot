"""Special-prop regression fixtures have known synthetic geometry, not game truth.

No mouse input, desktop capture or visible application is used by these tests.
"""
from dataclasses import replace
import math
from pathlib import Path
import queue
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytest

from calibration import PressModel
from controller import Runner, ReviewRequest, needs_review
from top_geometry import _quad, quad_center
from vision import Detection, DetectionError, annotate, detect, same_reviewed_frame, frame_difference
from test_learning import draw_piece, draw_platform


def textured_face(image, vertices, style="stripes"):
    p = np.asarray(vertices)
    top, right, bottom, left = [tuple(map(int, v)) for v in p]
    draw = ImageDraw.Draw(image)
    # Taller asymmetrical body/shadow deliberately changes the object's bbox.
    draw.polygon([(left[0]-45, left[1]+25), right, (bottom[0]-40, bottom[1]+90)], fill=(148, 148, 150))
    draw.polygon([left, bottom, right, (right[0], right[1]+74), (bottom[0], bottom[1]+74),
                  (left[0], left[1]+74)], fill=(88, 101, 105))
    mask = Image.new("L", image.size)
    ImageDraw.Draw(mask).polygon([top, right, bottom, left], fill=255)
    texture = Image.new("RGB", image.size, (221, 167, 70))
    paint = ImageDraw.Draw(texture)
    x0, y0 = p.min(axis=0).astype(int)
    x1, y1 = p.max(axis=0).astype(int)
    if style == "stripes":
        for x in range(x0-100, x1+100, 13):
            paint.line((x, y0-20, x+75, y1+20), fill=(86, 156, 158), width=6)
    elif style == "checker":
        for x in range(x0, x1+16, 16):
            for y in range(y0, y1+12, 12):
                if ((x-x0)//16+(y-y0)//12) % 2:
                    paint.rectangle((x,y,x+15,y+11), fill=(78, 143, 148))
    elif style == "logo":
        cx, cy = quad_center(p)
        paint.polygon([(cx+10,cy-25),(cx+48,cy-3),(cx+10,cy+19),(cx-28,cy-3)], fill=(44, 96, 103))
        paint.ellipse((cx+4,cy-14,cx+38,cy+6), fill=(242, 230, 198))
        paint.text((cx-34,cy+14), "SHOP 20", fill=(44, 64, 70))
    elif style == "raised":
        cx, cy = quad_center(p)
        paint.ellipse((cx+12,cy-26,cx+44,cy-10), fill=(55, 74, 82))
        paint.rectangle((cx+12,cy-39,cx+44,cy-18), fill=(115, 89, 37))
        paint.ellipse((cx+12,cy-46,cx+44,cy-30), fill=(244, 225, 132))
    elif style in ("pattern_logo", "center_logo"):
        for x in range(x0-100,x1+100,13):
            paint.line((x,y0-20,x+75,y1+20),fill=(86,156,158),width=6)
        cx,cy = quad_center(p)
        if style == "center_logo":
            cx -= 12
        # A complete, off-centre little diamond inside a fragmented outer rim.
        paint.polygon([(cx+12,cy-26),(cx+59,cy),(cx+12,cy+26),(cx-35,cy)],fill=(45,65,69))
    image.paste(texture, (0,0), mask)
    # A thin real rim is common on decorated objects; interior texture may touch
    # it, but it does not change any ground-truth corner coordinates.
    ImageDraw.Draw(image).line([top,right,bottom,left,top], fill=(45, 63, 69), width=2)


def prop_scene(style="stripes", current_style="plain", mirrored=False, foot_offset=(0,0)):
    image = Image.new("RGB", (540,965), (202,204,215))
    target = np.array([(380,370),(485,438),(403,493),(299,435)], np.float32)
    current = np.array([(160,565),(263,620),(160,675),(57,620)], np.float32)
    textured_face(image, target, style)
    textured_face(image, current, current_style)
    draw_piece(ImageDraw.Draw(image), (160+foot_offset[0],620+foot_offset[1]))
    expected_t = quad_center(target)
    expected_s = quad_center(current)
    if mirrored:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        expected_t = (539-expected_t[0],expected_t[1])
        expected_s = (539-expected_s[0],expected_s[1])
    return image, expected_s, expected_t


@pytest.mark.parametrize("style", ["stripes","checker","logo","raised"])
@pytest.mark.parametrize("current_style", ["plain","stripes","checker"])
@pytest.mark.parametrize("mirrored", [False,True])
def test_special_props_use_outer_top_centers_not_texture_or_whole_object(style,current_style,mirrored):
    image, current, target = prop_scene(style,current_style,mirrored)
    result = detect(image)
    assert math.dist(result.target,target) < 3
    assert result.current_center is not None
    assert math.dist(result.current_center,current) < 3
    assert result.confidence >= .75
    assert result.target_confidence >= .75 and result.current_confidence >= .84
    if "断边恢复" in result.note:
        assert result.confidence <= .79 and needs_review(result,.84)
    assert abs(result.center_distance-math.dist(current,target)) < 4


@pytest.mark.parametrize("scale", [.5,.75,1.5,2])
def test_prop_geometry_is_resolution_normalized(scale):
    image, s, t = prop_scene("checker", "stripes")
    resized = image.resize((round(image.width*scale),round(image.height*scale)), Image.Resampling.LANCZOS)
    result = detect(resized)
    actual_scale = resized.width/540
    assert math.dist(np.asarray(result.target)/actual_scale, t) < 3
    assert result.current_center is not None
    assert math.dist(np.asarray(result.current_center)/actual_scale, s) < 3


@pytest.mark.parametrize("offset", [(-20,10),(20,-10),(30,0)])
def test_current_top_center_does_not_follow_off_center_piece(offset):
    image, s, t = prop_scene("logo","stripes",foot_offset=offset)
    result = detect(image)
    assert result.current_center is not None and math.dist(result.current_center,s) < 3
    assert math.dist(result.piece,np.asarray(s)+offset) < 2
    assert math.dist(result.target,t) < 3
    assert abs(result.distance-math.dist(np.asarray(s)+offset,t)) < 3
    assert abs(result.center_distance-result.distance) > 8


@pytest.mark.parametrize("offset", [(-18,10),(0,0),(18,-10)])
def test_current_circular_top_ignores_piece_and_body(offset):
    image = Image.new("RGB",(540,965),(202,204,215))
    draw = ImageDraw.Draw(image)
    draw_platform(draw,(390,493),"diamond")
    draw_platform(draw,(170,620),"round",radius=(83,45))
    draw_piece(draw,(170+offset[0],620+offset[1]))
    result = detect(image)
    assert result.current_center is not None
    assert math.dist(result.current_center,(170,620)) < 3
    assert result.current_confidence >= .84


def test_perspective_center_is_diagonal_intersection_not_vertex_average():
    p = np.array([(175,360),(310,445),(205,485),(100,435)],np.float32)
    center = np.asarray(quad_center(p))
    assert math.dist(center,p.mean(axis=0)) > 7
    for start, end in ((p[0],p[2]),(p[1],p[3])):
        assert abs(np.cross(center-start,end-start)) < 1e-6


def test_perspective_quad_detection_uses_projective_center_end_to_end():
    image = Image.new("RGB",(540,965),(202,204,215))
    p = np.array([(175,360),(310,445),(205,485),(100,435)],np.float32)
    textured_face(image,p,"checker")
    draw = ImageDraw.Draw(image)
    draw_platform(draw,(420,610),"diamond")
    draw_piece(draw,(420,610))
    result = detect(image)
    assert math.dist(result.target,quad_center(p)) < 3
    assert math.dist(result.target,p.mean(axis=0)) > 5


@pytest.mark.parametrize("style", ["pattern_logo","center_logo"])
def test_inset_logo_cannot_hide_fragmented_outer_face(style):
    image,s,t = prop_scene(style,"plain")
    result = detect(image)
    assert math.dist(result.target,t) < 3
    assert result.confidence <= .79 and needs_review(result,.84)
    assert np.ptp(np.asarray(result.polygon)[:,0]) > 170


def test_decorated_round_prop_does_not_aim_at_offcenter_logo():
    image = Image.new("RGB",(540,965),(202,204,215))
    draw = ImageDraw.Draw(image)
    draw_platform(draw,(390,493),"round",radius=(83,45))
    draw.ellipse((320,454,460,532),outline=(216,183,115),width=4)
    draw.ellipse((399,479,449,507),fill=(212,174,113))
    draw.text((350,480),"PLAY",fill=(225,214,191))
    draw_platform(draw,(170,620),"diamond")
    draw_piece(draw,(170,620))
    result = detect(image)
    assert math.dist(result.target,(390,493)) < 2
    assert result.confidence >= .84


def test_severe_unsupported_occlusion_cannot_get_auto_confidence():
    image,s,t = prop_scene("checker")
    draw = ImageDraw.Draw(image)
    draw.rectangle((290,360,490,447),fill=(202,204,215))
    try:
        result = detect(image)
    except DetectionError:
        return
    assert result.confidence < .84


def test_only_next_prop_is_selected_with_more_upper_objects():
    image,s,t = prop_scene("logo")
    draw = ImageDraw.Draw(image)
    draw_platform(draw,(180,300),"round",radius=(65,36))
    result = detect(image)
    assert math.dist(result.target,t) < 3


def test_annotation_exposes_all_three_points_without_mutating_image():
    image,s,t = prop_scene("logo",foot_offset=(20,0))
    original = image.tobytes()
    result = detect(image)
    output = annotate(image,result,PressModel().predict(result.distance,result.width))
    assert original == image.tobytes() and output.size == image.size
    colors = set(output.getdata())
    assert {(74,222,128),(34,211,238),(251,191,36)}.issubset(colors)


@pytest.mark.parametrize("p", [
    [(300,450),(420,510),(420,575),(300,515)],  # vertical side
    [(300,450),(450,450),(450,530),(300,530)],  # label/button
    [(1,430),(101,375),(201,430),(101,485)],    # clipped
])
def test_side_faces_buttons_and_clipped_faces_are_not_tops(p):
    assert _quad(np.asarray(p,np.float32),np.zeros((965,540),np.float32)) is None


def test_current_missing_is_not_silently_replaced_by_foot():
    image = Image.new("RGB",(540,965),(202,204,215))
    draw = ImageDraw.Draw(image)
    draw_platform(draw,(390,493),"diamond")
    draw_piece(draw,(170,620))
    result = detect(image)
    assert result.current_center is None and result.center_distance is None
    assert result.confidence < .84 and needs_review(result,.84)


def test_manual_current_center_keeps_actual_foot_and_clears_approval():
    result = detect(prop_scene()[0]).with_confidence(.99).confirmed()
    changed = result.with_point("current_center", (154,621))
    assert changed.current_center == (154,621) and not changed.current_polygon
    assert changed.manual and not changed.human_confirmed and changed.confidence_override is None
    assert changed.distance == result.distance and changed.piece == result.piece
    assert changed.target == result.target and needs_review(changed,.84)
    assert changed.current_confidence == 0
    assert changed.with_point("piece",(170,610)).current_center is None
    for point in ((float("nan"),600),(540,620),(-1,620)):
        with pytest.raises(ValueError):
            result.with_point("current_center",point)


def test_review_checks_current_top_roi_even_when_far_from_piece():
    image = Image.new("RGB",(540,965),(202,204,215))
    result = Detection((150,640),(390,490),.95,540,965,current_center=(240,610))
    changed = image.copy()
    ImageDraw.Draw(changed).rectangle((230,600,249,619),fill=(30,40,50))
    assert frame_difference(image,changed) < 1
    assert same_reviewed_frame(image,image,result)
    assert not same_reviewed_frame(image,changed,result)


@pytest.mark.parametrize("center", [(float("inf"),620),(-1,620),(540,620),(160,965)])
def test_runner_rejects_invalid_reviewed_current_center(center):
    image,s,t = prop_scene()
    detection = detect(image)
    runner = Runner(queue.Queue())
    runner._pending = ReviewRequest(1,image,detection,"test",1,1)
    with pytest.raises(ValueError):
        runner.approve(1,replace(detection,current_center=center))
    assert not runner._review_event.is_set()


def test_gui_can_mark_current_top_center_in_review_without_input(monkeypatch,tmp_path,gui_root):
    import tkinter as tk
    import app as module
    monkeypatch.setattr(module,"APP_DIR",tmp_path)
    root = tk.Toplevel(gui_root)
    root.withdraw()
    app = module.App(root)
    try:
        image,s,t = prop_scene()
        app.show_frame(image)
        original = app.result
        app.image_transform = (1,0,0)
        app.mode.set("current_center")
        app.pick_point(SimpleNamespace(x=158,y=621))
        assert app.result.current_center == (158,621)
        assert app.result.target == original.target and app.result.piece == original.piece
        assert "顶心间距" in app.centers_text.get()
        assert app.result.manual and not app.armed.get() and not app.runner.running
        assert "1.4" in root.title()
    finally:
        app.close()
