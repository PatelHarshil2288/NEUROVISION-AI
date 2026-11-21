from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from neurovision.app.main_window import MainWindow


def main() -> int:
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

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
