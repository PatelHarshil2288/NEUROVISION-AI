from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path


def _ensure_stdio() -> None:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')


def main() -> int:
    _ensure_stdio()

    if '--version' in sys.argv:
        from neurovision import __version__

        print(__version__)
        return 0

    if '--diagnostics' in sys.argv:
        import torch
        from PySide6 import __version__ as pyside_version

        bundle_dir = getattr(sys, '_MEIPASS', None)
        checkpoint = (
            Path(bundle_dir) / 'data' / 'checkpoints' / 'mnist.pt'
            if bundle_dir
            else Path.cwd() / 'data' / 'checkpoints' / 'mnist.pt'
        )
        print(json.dumps({
            'platform': platform.platform(),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'pyside6': pyside_version,
            'checkpoint': str(checkpoint),
            'checkpoint_exists': checkpoint.is_file(),
        }, indent=2))
        return 0

    from PySide6.QtWidgets import QApplication

    from neurovision.app.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
