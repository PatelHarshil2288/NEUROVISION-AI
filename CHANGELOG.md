# Changelog

## 0.1.3

- Fixed MNIST download progress in windowed Windows builds when standard output streams are unavailable.
- Added regression coverage for initializing missing standard streams.

## 0.1.2

- Improved Digit Studio with single-click drawing and centered, aspect-preserving MNIST preprocessing.
- Added lightweight `--version` and JSON `--diagnostics` commands.
- Improved training progress with sample-weighted epoch loss and model-mode preservation.
- Added package metadata, community templates, policies, and automated dependency updates.
- Expanded the regression suite to cover preprocessing, GUI events, training, and CLI behavior.

## 0.1.1

- Bundled the trained MNIST checkpoint in source and Windows builds.
- Added SHA256 release verification.
- Added startup diagnostics mode.
- Added PyInstaller Windows release automation.

## 0.1.0

- Added the PySide6 desktop application shell.
- Added an MLP forward-pass implementation and telemetry.
- Added the 3D neural observatory.
- Added Digit Studio preprocessing and inference.
- Added MNIST training, checkpoint save/load, and CI.
