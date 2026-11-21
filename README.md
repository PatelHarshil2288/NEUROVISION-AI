# NEUROVISION AI

Interactive Neural Intelligence Laboratory (local-first).

## Current Status (Phases 1–4)

- PySide6 desktop shell launches.
- PyTorch MLP runs a real forward pass (784 -> 128 -> 64 -> 10).
- Telemetry is derived from real tensors.
- 3D viewport renders neurons + filtered connections (QOpenGLWidget + PyOpenGL).
- Digit Studio lets you draw a digit, preprocess it to 28x28, and run inference.
- Training Centre trains the model locally on MNIST and supports checkpoint save/load.

## Setup

1) Create and activate the conda env:

```bash
conda env create -f environment.yml
conda activate neurovision
```

2) Install the package (editable):

```bash
python -m pip install -e .
```

3) Run the app:

```bash
python -m neurovision
```

An installed package can also be launched with:

```bash
neurovision
```

Print local environment and checkpoint diagnostics with:

```bash
python -m neurovision --diagnostics
```

Print the installed package version with:

```bash
python -m neurovision --version
```

## Windows release

Push a version tag to build a standalone Windows executable:

```bash
git tag v0.1.1
git push origin v0.1.1
```

## Verify a Windows release

With both release files downloaded:

```powershell
.\scripts\verify_release.ps1 `
  -ZipPath .\NeurovisionAI-Windows.zip `
  -ChecksumJsonPath .\NeurovisionAI-Windows.zip.sha256.json
```

## License

NEUROVISION-AI is released under the MIT License. See [LICENSE](LICENSE).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and validation guidance.

## Code of Conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting guidance.

If `data/checkpoints/mnist.pt` exists, the app loads it automatically at startup.

The trained checkpoint is included in packaged releases and does not require a separate download.

4) Run tests:

```bash
pytest
```

## Controls (3D)

- Left-drag: orbit camera
- Mouse wheel: zoom
- Use "Play Propagation" after running inference

## Digit Studio

- Draw with left mouse button
- Click "Use Drawing For Inference" to feed it into the model

## Training Centre (MNIST)

- Click "Train on MNIST (CPU)" to download (first run) and train locally
- Checkpoints are standard PyTorch `.pt` files
- After training or loading a checkpoint, Digit Studio inference uses the updated model

### Offline notes

MNIST is cached under `./data/` after the first download, so later runs do not require internet.
