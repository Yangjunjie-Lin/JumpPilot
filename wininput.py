"""Windows-only screen binding and bounded, cancelable mouse holds.

No networking, process injection, game memory reading, or privilege escalation.
All screen/input actions are invoked only after explicit UI actions by the user.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
import threading
import time

from PIL import ImageGrab

IS_WINDOWS = os.name == "nt"
user32 = ctypes.WinDLL("user32", use_last_error=True) if IS_WINDOWS else None

if IS_WINDOWS:
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.WindowFromPoint.argtypes = [wintypes.POINT]
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]


class SafetyStop(RuntimeError):
    pass


def enable_dpi_awareness() -> None:
    if not IS_WINDOWS:
        return
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            user32.SetProcessDPIAware()


def _require_windows() -> None:
    if not IS_WINDOWS:
        raise SafetyStop("自动按压仅支持 Windows；其他系统可导入图片分析。")


def desktop_bounds() -> tuple[int, int, int, int]:
    _require_windows()
    return tuple(user32.GetSystemMetrics(index) for index in (76, 77, 78, 79))


def root_at(x: int, y: int) -> int:
    _require_windows()
    window = user32.WindowFromPoint(wintypes.POINT(round(x), round(y)))
    return int(user32.GetAncestor(window, 2) or window or 0)


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise SafetyStop("游戏窗口已关闭或无法读取。")
    return rect.left, rect.top, rect.right, rect.bottom


def process_id(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


@dataclass(frozen=True)
class Binding:
    hwnd: int
    pid: int
    title: str
    window_bounds: tuple[int, int, int, int]
    region: tuple[int, int, int, int]

    @classmethod
    def from_region(cls, region: tuple[int, int, int, int]) -> Binding:
        _require_windows()
        left, top, right, bottom = region
        if right - left < 150 or bottom - top < 250:
            raise SafetyStop("框选区域太小，请包含整个竖屏游戏画面。")
        hwnd = root_at((left + right) // 2, (top + bottom) // 2)
        if not hwnd or process_id(hwnd) == os.getpid():
            raise SafetyStop("请框选游戏，而不是本工具窗口。")
        bounds = window_rect(hwnd)
        if left < bounds[0] or top < bounds[1] or right > bounds[2] or bottom > bounds[3]:
            raise SafetyStop("框选区域必须完全位于同一个游戏窗口内。")
        title = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, title, 512)
        binding = cls(hwnd, process_id(hwnd), title.value, bounds, region)
        binding.verify(require_foreground=False)
        return binding

    def verify(self, *, require_foreground: bool = True, require_unobscured: bool = True) -> None:
        _require_windows()
        if (not user32.IsWindow(self.hwnd) or not user32.IsWindowVisible(self.hwnd)
                or user32.IsIconic(self.hwnd) or process_id(self.hwnd) != self.pid):
            raise SafetyStop("游戏窗口已关闭、隐藏或最小化。")
        if window_rect(self.hwnd) != self.window_bounds:
            raise SafetyStop("游戏窗口位置或大小改变，请重新框选绑定。")
        if require_foreground and int(user32.GetForegroundWindow() or 0) != self.hwnd:
            raise SafetyStop("焦点离开游戏窗口，已停止。")
        if not require_unobscured:
            return
        l, t, r, b = self.region
        # Screen capture is intentional, but obscured captures must never drive input.
        for fx, fy in ((.05, .05), (.95, .05), (.05, .95), (.95, .95), (.5, .5), (.5, .75)):
            if root_at(round(l + (r-l)*fx), round(t + (b-t)*fy)) != self.hwnd:
                raise SafetyStop("游戏画面被其他窗口遮挡，已停止。")

    def capture(self):
        self.verify(require_foreground=False)
        image = ImageGrab.grab(bbox=self.region, all_screens=True).convert("RGB")
        self.verify(require_foreground=False)
        return image

    def focus(self) -> None:
        # A review panel may cover the game; bring the bound game forward first.
        # Capture and input still require full foreground/occlusion checks.
        self.verify(require_foreground=False, require_unobscured=False)
        user32.SetForegroundWindow(self.hwnd)

    def screen_point(self, relative: tuple[float, float]) -> tuple[int, int]:
        l, t, r, b = self.region
        x, y = relative
        if not (.04 <= x <= .96 and .10 <= y <= .95):
            raise SafetyStop("按压点过于靠近边缘或标题栏；请设在游戏内空白处。")
        return round(l + x * (r-l)), round(t + y * (b-t))


def check_cancel(stop: threading.Event) -> None:
    if stop.is_set():
        raise SafetyStop("已停止。")
    _require_windows()
    if user32.GetAsyncKeyState(0x1B) & 0x8000:
        stop.set()
        raise SafetyStop("Esc 紧急停止。")
    cursor = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(cursor))
    vx, vy, _, _ = desktop_bounds()
    if cursor.x <= vx + 3 and cursor.y <= vy + 3:
        stop.set()
        raise SafetyStop("鼠标移至桌面左上角，触发安全停止。")


def pause(seconds: float, stop: threading.Event, binding: Binding | None = None) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check_cancel(stop)
        if binding is not None:
            binding.verify()
        stop.wait(min(.025, max(0, deadline-time.monotonic())))


if IS_WINDOWS:
    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("data",)
        _fields_ = [("type", wintypes.DWORD), ("data", INPUT_UNION)]

    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT


def _mouse(flags: int) -> None:
    event = INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, flags, 0, 0))
    if user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
        raise SafetyStop("系统拒绝鼠标输入；请确保游戏和工具权限一致，不要提升权限。")


def hold(binding: Binding, relative: tuple[float, float], duration_ms: int,
         stop: threading.Event) -> float:
    if not 40 <= duration_ms <= 2500:
        raise SafetyStop("按压时长超出硬性安全限制。")
    check_cancel(stop)
    binding.verify()
    if user32.GetAsyncKeyState(0x01) & 0x8000:
        raise SafetyStop("鼠标左键正被手动按住，请先松开。")
    point = binding.screen_point(relative)
    if root_at(*point) != binding.hwnd:
        raise SafetyStop("按压位置不属于绑定的游戏窗口。")
    original = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(original))
    if not user32.SetCursorPos(*point):
        raise SafetyStop("无法定位鼠标。")
    started = time.monotonic()
    down = False
    try:
        check_cancel(stop)
        binding.verify()
        _mouse(0x0002)
        down = True
        started = time.monotonic()
        pause(duration_ms / 1000, stop, binding)
    finally:
        if down:
            try:
                _mouse(0x0004)
            except SafetyStop:
                # Emergency release fallback; no second mouse-down is ever sent.
                user32.mouse_event(0x0004, 0, 0, 0, 0)
        cursor = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(cursor))
        if not stop.is_set() and (cursor.x, cursor.y) == point:
            user32.SetCursorPos(original.x, original.y)
    return (time.monotonic() - started) * 1000
