"""One Tcl interpreter per test process, matching the real app's lifecycle."""
import pytest


@pytest.fixture(scope="session")
def gui_root():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()
