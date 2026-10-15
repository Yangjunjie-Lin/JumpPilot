"""JumpPilot — local Windows visual jump assistant. Run without arguments for GUI."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import queue
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageGrab, ImageTk

from calibration import PressModel, Sample, fit_model
from controller import Runner, RunOptions, ReviewRequest
from score_reader import DEFAULT_SCORE_ROI, read_score, validate_roi
from vision import Detection, annotate, detect, load_image
import wininput

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
ASSETS = Path(getattr(sys, "_MEIPASS", APP_DIR))
BG = "#0b1120"
PANEL = "#121e30"
SURFACE = "#1b2a40"
TEXT = "#e6edf7"
MUTED = "#95a6bf"
CYAN = "#37d4de"
GOLD = "#f6c35c"


class RegionPicker(tk.Toplevel):
    def __init__(self, root: tk.Tk, callback):
        super().__init__(root)
        self.callback = callback
        self.vx, self.vy, self.vw, self.vh = wininput.desktop_bounds()
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.geometry(f"{self.vw}x{self.vh}{self.vx:+d}{self.vy:+d}")
        self.canvas = tk.Canvas(self, highlightthickness=0, cursor="crosshair", bg="#182235")
        self.canvas.pack(fill="both", expand=True)
        image = ImageGrab.grab(all_screens=True).convert("RGB")
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.create_rectangle(12, 12, 660, 86, fill=BG, outline=CYAN)
        self.canvas.create_text(30, 30, text="拖动框选完整的游戏画面，不要包含桌面或工具窗口", anchor="nw",
                                fill=TEXT, font=("Microsoft YaHei UI", 15, "bold"))
        self.canvas.create_text(30, 59, text="保留相同裁剪范围以保证距离标定一致  ·  Esc 取消", anchor="nw",
                                fill=CYAN, font=("Microsoft YaHei UI", 10))
        self.start = None
        self.rectangle = None
        self.canvas.bind("<ButtonPress-1>", self.down)
        self.canvas.bind("<B1-Motion>", self.move)
        self.canvas.bind("<ButtonRelease-1>", self.up)
        self.bind("<Escape>", lambda _: self.finish(None))
        self.focus_force()
        self.grab_set()

    def down(self, event):
        self.start = (event.x, event.y)
        if self.rectangle:
            self.canvas.delete(self.rectangle)
        self.rectangle = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                       outline=CYAN, width=4)

    def move(self, event):
        if self.start:
            self.canvas.coords(self.rectangle, *self.start, event.x, event.y)

    def up(self, event):
        if not self.start:
            return
        x1, y1 = self.start
        x2, y2 = event.x, event.y
        self.finish((min(x1, x2)+self.vx, min(y1, y2)+self.vy,
                     max(x1, x2)+self.vx, max(y1, y2)+self.vy))

    def finish(self, result):
        self.grab_release()
        self.destroy()
        self.callback(result)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JumpPilot 1.4 · 跳一跳视觉助手")
        root.configure(bg=BG)
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1170, screen_w-60)}x{min(940, screen_h-80)}+24+24")
        root.minsize(880, 650)
        self.events = queue.Queue()
        self.runner = Runner(self.events)
        self.image: Image.Image | None = None
        self.result: Detection | None = None
        self.binding: wininput.Binding | None = None
        self.source = ""
        self.click_point = (.5, .78)
        self.manual_points = {}
        self.samples: list[Sample] = []
        self.last_jump = None
        self.score_roi = DEFAULT_SCORE_ROI
        self.score_corner = None
        self.learning_baseline = None
        self.learning_history = []
        self.image_transform = (1, 0, 0)
        self.duration = None
        self.controls = []
        self.review_controls = []
        self.review_request: ReviewRequest | None = None
        self.busy = False
        self.closing = False
        self.mode = tk.StringVar(value="inspect")
        self.armed = tk.BooleanVar(value=False)
        self.k = tk.StringVar(value="2.7000")
        self.b = tk.StringVar(value="0.00")
        self.gain = tk.StringVar(value="1.000")
        self.count = tk.StringVar(value="5")
        self.settle = tk.StringVar(value="1.5")
        self.confidence_threshold = tk.StringVar(value="84")
        self.manual_score = tk.StringVar(value="")
        self.adaptive_enabled = tk.BooleanVar(value=False)
        self.game_score_text = tk.StringVar(value="游戏得分：未读取（区别于识别置信度）")
        self.learning_text = tk.StringVar(value="学习默认关闭。可靠加分 + 落点偏差连续一致 3 次后，小幅调整时长倍率。")
        self.review_text = tk.StringVar(value="人工评分与确认仅对当前帧有效，不会沿用到下一跳。")
        self.status = tk.StringVar(value="离线预览 · 不会自动按压")
        self.source_text = tk.StringVar(value="尚未绑定游戏窗口")
        self.detail = tk.StringVar(value="请先导入截图或框选游戏。")
        self.distance_text = tk.StringVar(value="—")
        self.centers_text = tk.StringVar(value="S 当前顶心 → T 下一顶心：未识别")
        self.time_text = tk.StringVar(value="—")
        self.score_text = tk.StringVar(value="—")
        self.sample_info = tk.StringVar(value="0 条样本 · 默认系数未经你的游戏验证")
        self._styles()
        self._build()
        self._load_config()
        for variable in (self.k, self.b, self.gain):
            variable.trace_add("write", self._model_changed)
        self.confidence_threshold.trace_add("write", lambda *_: self.armed.set(False))
        self.adaptive_enabled.trace_add("write", lambda *_: self.armed.set(False))
        root.bind("<Escape>", lambda _: self.stop())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(80, self._poll)
        sample = ASSETS / "sample.png"
        if sample.exists():
            self.open_image(sample)

    def _styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=TEXT, font=("Microsoft YaHei UI", 10))
        style.configure("TButton", background=SURFACE, foreground=TEXT, borderwidth=0,
                        padding=(12, 9), font=("Microsoft YaHei UI", 10))
        style.map("TButton", background=[("active", "#28405c"), ("disabled", "#172235")],
                  foreground=[("disabled", "#65758b")])
        style.configure("Accent.TButton", background=CYAN, foreground=BG, font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#73edf1"), ("disabled", "#274452")])
        style.configure("Stop.TButton", background="#77354d", foreground="#fff0f4")
        style.map("Stop.TButton", background=[("active", "#9c435e")])
        style.configure("TEntry", fieldbackground=SURFACE, foreground=TEXT, insertcolor=TEXT, padding=6)
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT, padding=5)
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure("TRadiobutton", background=PANEL, foreground=MUTED, padding=3)
        style.map("TRadiobutton", background=[("active", PANEL)], foreground=[("selected", CYAN)])
        style.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, foreground=TEXT,
                        rowheight=28, borderwidth=0)
        style.configure("Treeview.Heading", background="#24374f", foreground=TEXT)
        style.configure("TScrollbar", background=SURFACE, troughcolor=PANEL, borderwidth=0, arrowsize=11)

    def label(self, parent, text, color=MUTED, size=10, bold=False, **kwargs):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color,
                        font=("Microsoft YaHei UI", size, "bold" if bold else "normal"), **kwargs)

    def button(self, parent, text, callback, style="TButton"):
        button = ttk.Button(parent, text=text, command=lambda: self._guard(callback), style=style)
        self.controls.append(button)
        return button

    def entry(self, parent, variable, width=9):
        entry = ttk.Entry(parent, textvariable=variable, width=width, font=("Consolas", 11))
        self.controls.append(entry)
        return entry

    def _build(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 12))
        self.label(header, "JUMP / PILOT", CYAN, 22, True).pack(side="left")
        self.label(header, "跳一跳视觉助手", TEXT, 13).pack(side="left", padx=18)
        self.label(header, "LOCAL ONLY  ·  WINDOWS", MUTED, 9).pack(side="right")
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=20)
        body.columnconfigure(0, weight=1, minsize=320)
        body.columnconfigure(1, weight=0, minsize=480)
        body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=PANEL)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        preview_header = tk.Frame(left, bg=PANEL)
        preview_header.pack(fill="x", padx=16, pady=(14, 8))
        self.label(preview_header, "视觉定位", TEXT, 12, True).pack(side="left")
        self.label(preview_header, "S 当前顶心 · T 下一顶心 · A 落脚点", MUTED, 9).pack(side="right")
        self.canvas = tk.Canvas(left, bg="#080e19", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=12)
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.canvas.bind("<Button-1>", self.pick_point)
        tk.Label(left, textvariable=self.source_text, bg=PANEL, fg=MUTED,
                 anchor="w", font=("Microsoft YaHei UI", 9)).pack(fill="x", padx=16, pady=10)
        right_shell = tk.Frame(body, bg=PANEL, width=490)
        right_shell.grid(row=0, column=1, sticky="nsew")
        right_shell.grid_propagate(False)
        right_shell.rowconfigure(0, weight=1)
        right_shell.columnconfigure(0, weight=1)
        scroller = tk.Canvas(right_shell, bg=PANEL, highlightthickness=0, width=470)
        self.scroller = scroller
        scrollbar = ttk.Scrollbar(right_shell, orient="vertical", command=scroller.yview)
        scroller.configure(yscrollcommand=scrollbar.set)
        scroller.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        right = tk.Frame(scroller, bg=PANEL)
        content = scroller.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>", lambda _: scroller.configure(scrollregion=scroller.bbox("all")))
        scroller.bind("<Configure>", lambda e: scroller.itemconfigure(content, width=e.width))
        def scroll(event):
            if str(event.widget).startswith(str(right_shell)):
                scroller.yview_scroll(-int(event.delta / 120), "units")
        self.root.bind("<MouseWheel>", scroll)
        metrics = tk.Frame(right, bg=PANEL)
        metrics.pack(fill="x", padx=16, pady=(16, 12))
        for i, (title, var, color) in enumerate((("起跳 A→T / px", self.distance_text, CYAN),
                                                ("计算按压 / ms", self.time_text, GOLD),
                                                ("有效置信度", self.score_text, TEXT))):
            metrics.columnconfigure(i, weight=1)
            card = tk.Frame(metrics, bg=SURFACE)
            card.grid(row=0, column=i, sticky="nsew", padx=(0, 6 if i < 2 else 0))
            self.label(card, title, MUTED, 9).pack(anchor="w", padx=10, pady=(9, 0))
            tk.Label(card, textvariable=var, bg=SURFACE, fg=color, font=("Consolas", 22, "bold")).pack(anchor="w", padx=10, pady=(0, 9))
        tk.Label(right, textvariable=self.centers_text, bg=PANEL, fg="#4ade80", justify="left", wraplength=420,
                 font=("Microsoft YaHei UI", 10)).pack(fill="x", padx=18, pady=(0, 7))
        tk.Label(right, textvariable=self.detail, bg=PANEL, fg=MUTED, justify="left", wraplength=420,
                 font=("Microsoft YaHei UI", 9)).pack(fill="x", padx=18, pady=(0, 10))
        frame = self.section(right, "00", "置信度与人工确认")
        row = tk.Frame(frame, bg=PANEL)
        row.pack(fill="x")
        self.label(row, "自动阈值 %", size=9).pack(side="left")
        self.entry(row, self.confidence_threshold, 5).pack(side="left", padx=(5, 12))
        self.label(row, "本帧人工评分 %", size=9).pack(side="left")
        score_entry = self.entry(row, self.manual_score, 5)
        score_entry.pack(side="left", padx=5)
        self.review_controls.append(score_entry)
        row = tk.Frame(frame, bg=PANEL)
        row.pack(fill="x", pady=(8, 4))
        for title, callback, style in (("应用本帧评分", self.apply_score, "TButton"),
                                       ("恢复算法评分", self.clear_score, "TButton"),
                                       ("确认本帧 / 继续", self.confirm_frame, "Accent.TButton")):
            is_confirmation = title.startswith("确认")
            button = ttk.Button(frame if is_confirmation else row, text=title, style=style,
                                command=lambda action=callback: self._review_guard(action))
            if is_confirmation:
                button.pack(fill="x", pady=(4, 2))
            else:
                button.pack(side="left", expand=True, fill="x", padx=(0, 5))
            self.controls.append(button)
            self.review_controls.append(button)
        tk.Label(frame, textvariable=self.review_text, bg=PANEL, fg=GOLD, wraplength=420,
                 justify="left", font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(4, 0))
        frame = self.section(right, "01", "画面与标记")
        row = tk.Frame(frame, bg=PANEL)
        row.pack(fill="x")
        self.button(row, "导入图片", self.open_image).pack(side="left", expand=True, fill="x", padx=(0, 7))
        self.button(row, "框选游戏", self.select_region, "Accent.TButton").pack(side="left", expand=True, fill="x", padx=(0, 7))
        self.button(row, "刷新识别", self.refresh).pack(side="left", expand=True, fill="x")
        modes = tk.Frame(frame, bg=PANEL)
        modes.pack(fill="x", pady=(8, 2))
        for index, (text, value) in enumerate((("查看", "inspect"), ("标落脚 A", "piece"),
                                               ("标当前顶心 S", "current_center"), ("标下一顶心 T", "target"),
                                               ("设按压点", "click"))):
            radio = ttk.Radiobutton(modes, text=text, value=value, variable=self.mode)
            radio.grid(row=index//3, column=index%3, sticky="w", padx=(0, 8))
            self.controls.append(radio)
            if value != "click":
                self.review_controls.append(radio)
        self.label(frame, "标的是可落脚顶面的中心，不是整个道具或文字中心。\n选模式后点左图；起跳计算使用 A→T，绿色 S→T 供核对。", size=9,
                   wraplength=420, justify="left").pack(anchor="w")
        self.click_text = tk.StringVar(value="按压点：画面 50% × 78%（请核对）")
        tk.Label(frame, textvariable=self.click_text, bg=PANEL, fg=CYAN, font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(2, 0))
        frame = self.section(right, "02", "距离 → 按压时长")
        self.label(frame, "t = 微调 × ( k × d标准 + b )", TEXT, 12, True).pack(anchor="w")
        self.label(frame, "d标准 = 屏幕距离 × 540 / 游戏画面宽度", size=9).pack(anchor="w", pady=(2, 8))
        row = tk.Frame(frame, bg=PANEL)
        row.pack(fill="x")
        for label, var in (("k · ms/px", self.k), ("b · ms", self.b), ("时长微调", self.gain)):
            cell = tk.Frame(row, bg=PANEL)
            cell.pack(side="left", expand=True, fill="x", padx=(0, 8))
            self.label(cell, label, size=9).pack(anchor="w")
            self.entry(cell, var).pack(fill="x", pady=(3, 0))
        self.label(frame, "跳短：小幅增大 k；跳远：小幅减小 k。微调不是物理压力。", GOLD, 9,
                   wraplength=420, justify="left").pack(anchor="w", pady=(8, 0))
        frame = self.section(right, "03", "成功样本校准")
        tk.Label(frame, textvariable=self.sample_info, bg=PANEL, fg=MUTED, justify="left", wraplength=420,
                 font=("Microsoft YaHei UI", 9)).pack(anchor="w")
        row = tk.Frame(frame, bg=PANEL)
        row.pack(fill="x", pady=(7, 0))
        self.button(row, "添加成功样本", self.add_sample_dialog).pack(side="left", padx=(0, 6))
        self.button(row, "拟合 k / b", self.fit).pack(side="left", padx=(0, 6))
        self.button(row, "查看 / 删除", self.show_samples).pack(side="left")
        row = tk.Frame(frame, bg=PANEL)
        row.pack(fill="x", pady=(7, 0))
        self.button(row, "保存参数", self.save_config).pack(side="left", padx=(0, 6))
        self.button(row, "导出标记图", self.export_image).pack(side="left")
        frame = self.section(right, "03+", "得分与落点自适应校准")
        checkbox = ttk.Checkbutton(frame, text="启用本轮学习（连续可靠反馈后调整时长倍率）", variable=self.adaptive_enabled)
        checkbox.pack(anchor="w")
        self.controls.append(checkbox)
        tk.Label(frame, textvariable=self.game_score_text, bg=PANEL, fg=CYAN, wraplength=420,
                 justify="left", font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=4)
        row = tk.Frame(frame, bg=PANEL)
        row.pack(fill="x")
        self.button(row, "框选分数", self.begin_score_region).pack(side="left", padx=(0, 5))
        self.button(row, "测试读分", self.update_game_score).pack(side="left", padx=(0, 5))
        self.button(row, "撤销本轮学习", self.undo_learning).pack(side="left")
        tk.Label(frame, textvariable=self.learning_text, bg=PANEL, fg=GOLD, wraplength=420,
                 justify="left", font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=5)
        self.label(frame, "分数框：在左图依次点左上/右下角，只包含游戏数字。\n"
                   "偏短增大时长，偏远减小时长；每次最多 2%，本轮最多 ±15%。\n"
                   "读分/落点不可靠就不学习；k、b 保持不变，保存参数后才持久化。", size=9,
                   wraplength=420, justify="left").pack(anchor="w")
        frame = self.section(right, "04", "单步验证，再自动运行")
        row = tk.Frame(frame, bg=PANEL)
        row.pack(fill="x")
        self.label(row, "本轮步数").pack(side="left")
        self.entry(row, self.count, 5).pack(side="left", padx=(6, 16))
        self.label(row, "落地等待 / s").pack(side="left")
        self.entry(row, self.settle, 5).pack(side="left", padx=6)
        self.label(frame, "0 = 无限跳；正整数 = 指定次数，无 30 步上限。", CYAN, 9,
                   wraplength=420, justify="left").pack(anchor="w", pady=(7, 0))
        checkbox = ttk.Checkbutton(frame, text="已校准参数，并核对游戏区域和按压点", variable=self.armed)
        checkbox.pack(anchor="w", pady=(9, 4))
        self.controls.append(checkbox)
        row = tk.Frame(frame, bg=PANEL)
        row.pack(fill="x")
        self.button(row, "单步试跳", lambda: self.start(1)).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.button(row, "开始自动跳", lambda: self.start(None), "Accent.TButton").pack(side="left", fill="x", expand=True)
        self.label(frame, "启动有 3 秒倒计时。Esc / 停止按钮 / 鼠标移至桌面左上角停止。", GOLD, 9,
                   wraplength=420, justify="left").pack(anchor="w", pady=(9, 0))
        self.label(frame, "低置信度会暂停，允许修正 S/T/A 后确认本跳。\n离线图片不能按压；启动前手动标点只允许单步。", MUTED, 9,
                   wraplength=420, justify="left").pack(anchor="w", pady=(7, 14))
        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=20, pady=12)
        self.stop_button = ttk.Button(footer, text="■  停止 / Esc", style="Stop.TButton", command=self.stop)
        self.stop_button.pack(side="right", padx=(12, 0))
        tk.Label(footer, textvariable=self.status, bg=BG, fg=CYAN, anchor="w", wraplength=850,
                 font=("Microsoft YaHei UI", 10)).pack(side="left", fill="x", expand=True)

    def section(self, parent, number, title):
        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(fill="x", padx=18, pady=(7, 10))
        self.label(frame, f"{number}   {title}", TEXT, 11, True).pack(anchor="w", pady=(0, 8))
        return frame

    def _guard(self, callback):
        if self.busy or self.runner.running:
            return
        try:
            callback()
        except Exception as error:
            self.status.set(str(error))
            messagebox.showerror("未执行操作", str(error), parent=self.root)

    def model(self):
        model = PressModel(float(self.k.get()), float(self.b.get()), float(self.gain.get()))
        model.validate()
        return model

    def _model_changed(self, *_):
        self.armed.set(False)
        self.update_metrics()

    def begin_score_region(self):
        if self.image is None:
            raise ValueError("请先导入或截图游戏画面。")
        self.armed.set(False)
        self.score_corner = None
        self.mode.set("score_roi")
        self.status.set("在预览图点击分数区域的左上角，然后点击右下角；仅框住数字并留少量边距。")

    def update_game_score(self):
        if self.image is None:
            return
        reading = read_score(self.image, self.score_roi)
        if reading.value is None:
            self.game_score_text.set(f"游戏得分：不可用 · {reading.reason}")
        else:
            self.game_score_text.set(f"游戏得分：{reading.value} · 读分质量 {reading.confidence:.0%}（非落点置信度）")

    def undo_learning(self):
        if self.learning_baseline is None:
            raise ValueError("尚无本轮学习基线可恢复。")
        model = self.learning_baseline
        self.gain.set(str(model.gain))
        self.adaptive_enabled.set(False)
        self.learning_text.set("已恢复本轮启动前的时长倍率，并关闭学习；k/b 保持当前值，需要时再保存。")
        self.armed.set(False)

    def threshold(self) -> float:
        try:
            value = float(self.confidence_threshold.get()) / 100
            RunOptions(self.model(), self.click_point, minimum_confidence=value).validate()
            return value
        except (ValueError, OverflowError):
            raise ValueError("自动放行阈值必须是 0–100 之间的有限数字。")

    def _review_guard(self, callback):
        if (self.busy or self.runner.running) and self.review_request is None:
            return
        try:
            callback()
        except Exception as error:
            self.status.set(str(error))
            messagebox.showerror("未执行操作", str(error), parent=self.root)

    def apply_score(self):
        if self.result is None:
            raise ValueError("当前没有有效点位，不能通过改分数生成识别结果。")
        try:
            score = float(self.manual_score.get()) / 100
        except (ValueError, OverflowError):
            raise ValueError("请填写 0–100 之间的人工置信度。")
        self.result = self.result.with_confidence(score)
        self.armed.set(False)
        self.update_metrics()
        self.review_text.set(f"算法 {self.result.confidence:.0%} → 本帧人工 {score:.0%}；"
                             + ("请点击确认本帧 / 继续。" if self.review_request else "仅作用当前帧。"))

    def clear_score(self):
        if self.result is not None:
            self.result = self.result.with_confidence(None)
        self.manual_score.set("")
        self.armed.set(False)
        self.update_metrics()
        self.review_text.set("已恢复算法评分；仍可人工确认当前帧。")

    def confirm_frame(self):
        if self.result is None:
            raise ValueError("没有可确认的点位，请先标记落脚 A 和下一顶心 T。")
        if self.manual_score.get().strip():
            self.result = self.result.with_confidence(float(self.manual_score.get()) / 100)
        # Check the press model before accepting even a manually approved frame.
        self.model().predict(self.result.distance, self.result.width)
        confirmed = self.result.confirmed()
        if self.review_request is not None:
            self.runner.approve(self.review_request.request_id, confirmed)
            self.review_request = None
            self._set_busy(True)
            self.status.set("已人工确认本帧，正在重新聚焦游戏；画面改变则确认作废。")
            self.review_text.set("本帧已确认 · 倒计时后只放行这一跳")
        else:
            self.review_text.set("本帧已人工确认；仍需勾选运行确认并启动。画面变化会使确认作废。")
            self.status.set("人工确认不会立即按压；请按正常流程选择单步或自动跳。")
        self.result = confirmed
        self.update_metrics()

    def update_metrics(self):
        result = self.result
        self.duration = None
        if result is None:
            self.distance_text.set("—")
            self.time_text.set("—")
            self.score_text.set("—")
            self.centers_text.set("S 当前顶心 → T 下一顶心：未识别")
        else:
            self.distance_text.set(f"{result.distance:.1f}")
            self.score_text.set(f"{result.effective_confidence:.0%}")
            provenance = f"算法 {result.confidence:.0%}"
            if result.confidence_override is not None:
                provenance += f" / 人工 {result.confidence_override:.0%}（仅本帧）"
            if result.manual:
                provenance += " / 坐标已手改"
            if result.human_confirmed:
                provenance += " / 已人工确认"
            if result.current_center is not None:
                sx, sy = result.current_center
                self.centers_text.set(f"顶心 S ({sx:.1f}, {sy:.1f}) → T ({result.target[0]:.1f}, {result.target[1]:.1f})\n"
                                      f"顶心间距 {result.center_distance:.1f} px · 落脚偏离 S {math.dist(result.piece, result.current_center):.1f} px")
            else:
                self.centers_text.set("当前顶心 S 未识别：可选“标当前顶心 S”手动修正。")
            self.detail.set(f"实际落脚 A ({result.piece[0]:.0f}, {result.piece[1]:.0f})  →  下一顶心 T ({result.target[0]:.0f}, {result.target[1]:.0f})\n"
                            f"Δx {result.dx:+.1f}   Δy {result.dy:+.1f}   标准距离 {result.normalized_distance:.1f} px\n"
                            f"当前顶面评分 {result.current_confidence:.0%} · 下一顶面评分 {result.target_confidence:.0%}\n"
                            f"{provenance}\n{result.note}")
            try:
                self.duration = self.model().predict(result.distance, result.width)
                self.time_text.set(str(self.duration))
            except (ValueError, OverflowError):
                self.time_text.set("无效")
        self.draw()

    def show_frame(self, image, result=None, *, analyse=True):
        self.image = image
        self.manual_points = {}
        self.result = None
        if analyse:
            try:
                result = detect(image)
            except ValueError as error:
                self.detail.set(str(error))
                self.status.set(str(error))
        self.result = result
        self.update_game_score()
        self.manual_score.set("" if result is None or result.confidence_override is None else
                              f"{result.confidence_override*100:.1f}")
        if self.review_request is None:
            self.review_text.set("人工评分与确认仅对当前帧有效；低于阈值将暂停等待确认。")
        self.update_metrics()

    def open_image(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(title="选择游戏截图", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if not path:
            return
        image = load_image(path)
        self.binding = None
        self.armed.set(False)
        self.source = "file"
        self.source_text.set(f"离线图片 · {image.width} × {image.height} · 不会触发按压")
        self.status.set("离线预览：默认时长未经游戏校准；可框选游戏进行试跳。")
        self.show_frame(image)

    def select_region(self):
        if not wininput.IS_WINDOWS:
            raise ValueError("屏幕控制仅支持 Windows。")
        self.armed.set(False)
        self.binding = None
        self.source = "file"
        self.source_text.set("尚未绑定游戏窗口 · 自动按压关闭")
        self.root.withdraw()
        self.root.after(350, self._open_picker)

    def _open_picker(self):
        try:
            RegionPicker(self.root, self._picked)
        except Exception as error:
            self.root.deiconify()
            messagebox.showerror("无法框选", str(error))

    def _picked(self, region):
        # Wait until the overlay is gone before resolving the underlying window.
        self.root.after(180, lambda: self._finish_selection(region))

    def _finish_selection(self, region):
        try:
            if region is None:
                self.status.set("已取消框选 · 自动按压关闭")
                return
            binding = wininput.Binding.from_region(region)
            image = binding.capture()
            self.binding = binding
            self.source = "screen"
            self.last_jump = None
            self.score_roi = DEFAULT_SCORE_ROI
            self.score_corner = None
            self.adaptive_enabled.set(False)
            self.click_point = (.5, .78)
            self.click_text.set("按压点：画面 50% × 78%（请核对）")
            self.source_text.set(f"已绑定 · {binding.title[:26]} · {image.width} × {image.height}")
            self.status.set("已绑定游戏。请把本工具移到游戏旁边；确认标记、参数和按压点，再单步试跳。")
            self.show_frame(image)
        except Exception as error:
            self.binding = None
            self.source = "file"
            self.status.set(str(error))
            messagebox.showerror("绑定失败", str(error))
        finally:
            self.root.deiconify()

    def refresh(self):
        self.armed.set(False)
        if self.binding:
            self.show_frame(self.binding.capture())
            self.status.set("屏幕已刷新。请核对当前标记。")
        elif self.image:
            self.show_frame(self.image)

    def draw(self):
        if self.image is None or not hasattr(self, "canvas"):
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 5 or ch < 5:
            return
        rendered = annotate(self.image, self.result, self.duration) if self.result else self.image
        scale = min((cw-14)/rendered.width, (ch-14)/rendered.height)
        width, height = max(1, round(rendered.width*scale)), max(1, round(rendered.height*scale))
        resized = rendered.resize((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        ox, oy = (cw-width)/2, (ch-height)/2
        self.image_transform = (scale, ox, oy)
        self.canvas.create_image(ox, oy, image=self.photo, anchor="nw")
        x1, y1, x2, y2 = self.score_roi
        self.canvas.create_rectangle(ox+x1*self.image.width*scale, oy+y1*self.image.height*scale,
                                     ox+x2*self.image.width*scale, oy+y2*self.image.height*scale,
                                     outline="#a78bfa", width=1, dash=(3, 3))
        for name, point in self.manual_points.items():
            x, y = ox + point[0]*scale, oy + point[1]*scale
            color = {"piece": CYAN, "current_center": "#4ade80", "target": GOLD}.get(name, GOLD)
            self.canvas.create_oval(x-6, y-6, x+6, y+6, outline=color, width=3)
        if self.binding:
            x = ox+self.click_point[0]*self.image.width*scale
            y = oy+self.click_point[1]*self.image.height*scale
            self.canvas.create_oval(x-8, y-8, x+8, y+8, outline="#ffffff", width=2, dash=(3, 3))
            self.canvas.create_text(x, y+20, text="按压", fill="#ffffff", font=("Microsoft YaHei UI", 9))

    def pick_point(self, event):
        if ((self.busy or self.runner.running) and self.review_request is None) or self.image is None or self.mode.get() == "inspect":
            return
        scale, ox, oy = self.image_transform
        point = ((event.x-ox)/scale, (event.y-oy)/scale)
        if not (0 <= point[0] < self.image.width and 0 <= point[1] < self.image.height):
            return
        self.armed.set(False)
        mode = self.mode.get()
        if mode == "score_roi":
            if self.review_request is not None:
                return
            if self.score_corner is None:
                self.score_corner = point
                self.status.set("再点击分数区域右下角；数字必须完整，留少量边距。")
            else:
                first = self.score_corner
                roi = (min(first[0], point[0])/self.image.width, min(first[1], point[1])/self.image.height,
                       max(first[0], point[0])/self.image.width, max(first[1], point[1])/self.image.height)
                try:
                    validate_roi(roi)
                except ValueError as error:
                    self.score_corner = None
                    self.status.set(str(error)+" 请重新点击两个角。")
                    return
                self.score_roi = roi
                self.score_corner = None
                self.mode.set("inspect")
                self.update_game_score()
                self.status.set("分数区域已设置；请先测试读分，再启用本轮学习。")
            self.draw()
            return
        elif mode == "click":
            if self.review_request is not None:
                return
            relative = (point[0]/self.image.width, point[1]/self.image.height)
            if not (.04 <= relative[0] <= .96 and .1 <= relative[1] <= .95):
                self.status.set("按压点不可位于边缘或标题栏，请选择游戏内空白处。")
                return
            self.click_point = relative
            self.click_text.set(f"按压点：画面 {relative[0]:.1%} × {relative[1]:.1%}")
        elif self.result:
            self.result = self.result.with_point(mode, point)
            self.manual_score.set("")
        else:
            self.manual_points[mode] = point
            if "piece" in self.manual_points and "target" in self.manual_points:
                self.result = Detection(self.manual_points["piece"], self.manual_points["target"], 0,
                                        self.image.width, self.image.height, manual=True,
                                        current_center=self.manual_points.get("current_center"),
                                        note="手动标点；仅允许单步试跳")
        self.update_metrics()

    def start(self, count):
        if not self.binding or self.source != "screen":
            raise ValueError("当前是离线图片。请先用“框选游戏”绑定真实游戏窗口。")
        if not self.armed.get():
            raise ValueError("请先校准或核对参数，并勾选确认区域和按压点。")
        if self.result is None:
            raise ValueError("当前没有有效识别；请刷新或手动标记两个点。")
        try:
            steps = int(self.count.get()) if count is None else count
        except ValueError:
            raise ValueError("本轮步数必须是非负整数；填写 0 可无限跳。") from None
        if self.result.manual and count is None:
            raise ValueError("手动标点不能连续自动跳，请使用单步试跳。")
        options = RunOptions(self.model(), self.click_point, steps, float(self.settle.get()), self.threshold(),
                             self.adaptive_enabled.get(), self.score_roi)
        options.validate()
        self.binding.verify(require_foreground=False, require_unobscured=False)
        self.binding.screen_point(self.click_point)
        self.learning_history.clear()
        self.learning_baseline = options.model if options.adaptive_learning else None
        if options.adaptive_learning:
            self.learning_text.set("本轮学习已启用：等待稳定加分与可靠落点，未收集足够一致反馈前不改参。")
        else:
            self.learning_text.set("本轮学习关闭，时长倍率保持不变。")
        use_current = self.result.manual or self.result.confidence_override is not None or self.result.human_confirmed
        self.runner.start(self.binding, options,
                          self.result if use_current else None,
                          self.image if use_current else None)
        self._set_busy(True)

    def _set_busy(self, busy):
        self.busy = busy
        for control in self.controls:
            if control.winfo_exists():
                control.configure(state="disabled" if busy else "normal")
        if busy and self.review_request is not None:
            for control in self.review_controls:
                control.configure(state="normal")

    def stop(self):
        self.runner.stop()
        self.review_request = None
        if self.runner.running:
            self._set_busy(True)
        self.armed.set(False)
        self.status.set("正在停止并释放鼠标…" if self.runner.running else "已停止 · 自动按压关闭")

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.set(value)
                elif kind == "review":
                    if self.runner.stop_event.is_set():
                        continue
                    self.show_frame(value.image, value.detection, analyse=False)
                    self.review_request = value
                    self.mode.set("inspect")
                    self._set_busy(True)
                    total_label = "∞" if value.total == 0 else str(value.total)
                    self.review_text.set(f"第 {value.step}/{total_label} 跳待确认：{value.reason}\n"
                                         "核对 S/T/A，可改分数或点位，然后点“确认本帧 / 继续”。")
                    self.status.set("已暂停，未按压 · 等待你人工确认当前帧")
                    self.scroller.yview_moveto(0)
                    self.root.deiconify()
                    self.root.lift()
                    self.root.focus_force()
                elif kind == "review_closed":
                    if self.review_request is not None and self.review_request.request_id == value:
                        self.review_request = None
                    self._set_busy(True)
                elif kind == "preview":
                    image, result, _ = value
                    self.show_frame(image, result, analyse=False)
                elif kind == "jump":
                    self.last_jump = value
                elif kind == "learning":
                    self.learning_history.append(value)
                    self.learning_history[:] = self.learning_history[-100:]
                    self.learning_text.set(f"第 {value['step']} 跳 · {value['message']}\n"
                                           f"有效反馈 {value['accepted']} 次 / 参数调整 {value['updates']} 次")
                    if value["changed"]:
                        self.gain.set(f"{value['model'].gain:.8f}")
                elif kind == "done":
                    self.review_request = None
                    count, reason = value
                    self.status.set(f"已停止 · 执行 {count} 次按压（不等于成功次数） · {reason}")
                    self.armed.set(False)
                    self._set_busy(False)
        except queue.Empty:
            pass
        if self.closing and not self.runner.running:
            self.root.destroy()
            return
        self.root.after(80, self._poll)

    def add_sample_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("添加已确认成功的校准样本")
        dialog.configure(bg=PANEL)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        self.label(dialog, "仅记录你确认落在平台中心附近的一跳", TEXT, 12, True).pack(padx=20, pady=(20, 8))
        if self.last_jump:
            defaults = (self.last_jump["distance"], self.last_jump["width"], self.last_jump["duration"])
            explanation = "已填入最近一次实际按压对应的跳前距离。请核对落地结果。"
        else:
            defaults = (self.result.distance if self.result else 0, self.image.width if self.image else 540, "")
            explanation = "时长必须来自你已验证的跳跃；不要把程序的预测当作成功样本。"
        self.label(dialog, explanation, GOLD, 9, wraplength=440, justify="left").pack(padx=20, pady=(0, 8))
        variables = []
        for title, value in zip(("跳前两点距离 / px", "当时游戏画面宽度 / px", "成功按压时长 / ms"), defaults):
            row = tk.Frame(dialog, bg=PANEL)
            row.pack(fill="x", padx=20, pady=5)
            self.label(row, title).pack(side="left")
            var = tk.StringVar(value=f"{value:.2f}" if isinstance(value, float) else str(value))
            ttk.Entry(row, textvariable=var, width=14).pack(side="right")
            variables.append(var)
        confirmed = tk.BooleanVar(value=False)
        ttk.Checkbutton(dialog, text="我确认这次跳跃成功，并落在平台中心附近", variable=confirmed).pack(padx=16, pady=8)
        def save():
            try:
                if not confirmed.get():
                    raise ValueError("请先确认这是一条实际成功的跳跃样本。")
                sample = Sample(float(variables[0].get()), int(float(variables[1].get())), float(variables[2].get()))
                sample.validate()
                self.samples.append(sample)
                self.sample_info.set(f"{len(self.samples)} 条成功样本 · 请拟合后再保存参数")
                self.armed.set(False)
                dialog.destroy()
            except ValueError as error:
                messagebox.showerror("样本无效", str(error), parent=dialog)
        ttk.Button(dialog, text="添加成功样本", command=save, style="Accent.TButton").pack(pady=(4, 18))
        dialog.grab_set()

    def fit(self):
        model, rmse = fit_model(self.samples)
        self.k.set(f"{model.k:.5f}")
        self.b.set(f"{model.b:.3f}")
        self.gain.set("1.000")
        self.sample_info.set(f"{len(self.samples)} 条样本 · 拟合 RMSE {rmse:.1f} ms（非成功率）")
        self.status.set("已拟合。两点拟合只是初始标定，建议使用 3–5 条不同距离样本，并单步验证。")

    def show_samples(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("校准样本")
        dialog.configure(bg=PANEL)
        dialog.transient(self.root)
        table = ttk.Treeview(dialog, columns=("distance", "width", "duration"), show="headings", height=8)
        for key, label in (("distance", "跳前距离 / px"), ("width", "截图宽度 / px"), ("duration", "按压时长 / ms")):
            table.heading(key, text=label)
            table.column(key, width=140, anchor="center")
        def populate():
            for child in table.get_children():
                table.delete(child)
            for i, sample in enumerate(self.samples):
                table.insert("", "end", iid=str(i), values=(f"{sample.distance_px:.1f}", sample.width, f"{sample.duration_ms:.0f}"))
        populate()
        table.pack(padx=16, pady=16)
        def delete_selected():
            for index in sorted((int(i) for i in table.selection()), reverse=True):
                self.samples.pop(index)
            populate()
            self.sample_info.set(f"{len(self.samples)} 条样本 · 样本已更改，请重新拟合")
            self.armed.set(False)
        ttk.Button(dialog, text="删除选中样本", command=delete_selected).pack(pady=(0, 16))
        dialog.grab_set()

    def save_config(self):
        model = self.model()
        data = {"version": 1, "model": model.to_dict(), "samples": [asdict(s) for s in self.samples],
                "minimum_confidence": self.threshold(), "score_roi": list(self.score_roi)}
        destination = APP_DIR / "jump-pilot-settings.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
        self.status.set("已保存参数、样本、阈值和分数区域；人工确认、绑定、运行与学习开关不会保存。")

    def _load_config(self):
        path = APP_DIR / "jump-pilot-settings.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                raise ValueError("参数版本不支持")
            model = PressModel(**data["model"])
            model.validate()
            samples = [Sample(**s) for s in data.get("samples", [])[:200]]
            for sample in samples:
                sample.validate()
            confidence = data.get("minimum_confidence", .84)
            RunOptions(model, self.click_point, minimum_confidence=confidence).validate()
            score_roi = tuple(data.get("score_roi", DEFAULT_SCORE_ROI))
            validate_roi(score_roi)
            self.k.set(str(model.k))
            self.b.set(str(model.b))
            self.gain.set(str(model.gain))
            self.confidence_threshold.set(f"{confidence*100:g}")
            self.samples = samples
            self.score_roi = score_roi
            self.sample_info.set(f"已载入 {len(samples)} 条样本和本地参数 · 启动时自动按压始终关闭")
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.status.set(f"参数文件无效，已使用默认值：{error}")

    def export_image(self):
        if self.image is None or self.result is None:
            raise ValueError("请先识别或标记两点。")
        path = filedialog.asksaveasfilename(title="导出标记图", defaultextension=".png", initialfile="jump-analysis.png", filetypes=[("PNG 图片", "*.png")])
        if path:
            annotate(self.image, self.result, self.duration).save(path)
            self.status.set("标记图已导出。")

    def close(self):
        self.closing = True
        self.stop()
        if not self.runner.running:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="JumpPilot visual jump assistant")
    parser.add_argument("--analyze", type=Path, help="Offline image analysis; never performs input")
    parser.add_argument("--output", type=Path, help="Save annotated PNG for --analyze")
    parser.add_argument("--report", type=Path, help="Save offline analysis JSON, including score reading")
    parser.add_argument("--smoke-test", action="store_true", help="Create the GUI with offline sample and close; no screen control")
    args = parser.parse_args()
    if args.report and not args.analyze:
        parser.error("--report requires --analyze")
    if args.analyze:
        image = load_image(args.analyze)
        result = detect(image)
        duration = PressModel().predict(result.distance, image.width)
        score = read_score(image)
        report = {"piece": result.piece, "target": result.target, "distance_px": result.distance,
                  "current_center": result.current_center, "center_distance_px": result.center_distance,
                  "current_polygon": result.current_polygon, "target_polygon": result.polygon,
                  "current_confidence": result.current_confidence, "target_confidence": result.target_confidence,
                  "press_distance_policy": "actual_foot_to_target_top_center", "note": result.note,
                  "normalized_distance": result.normalized_distance, "score": result.confidence,
                  "game_score": score.value, "game_score_confidence": score.confidence,
                  "target_policy": "nearest_upper_row", "example_duration_ms": duration, "calibrated": False}
        rendered_report = json.dumps(report, indent=2)
        print(rendered_report)
        if args.report:
            args.report.write_text(rendered_report, encoding="utf-8")
        if args.output:
            annotate(image, result, duration).save(args.output)
        return
    wininput.enable_dpi_awareness()
    root = tk.Tk()
    app = App(root)
    if args.smoke_test:
        root.after(700, app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
