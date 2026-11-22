# NEUROVISION AI Frontend

The Next.js App Router frontend includes Digit Studio, the Neural Observatory, the local Training Lab, the Activation and Weight Explorer, and Model Analytics through Phase 8. Digit Studio sends processed 28×28 grayscale pixels to versioned inference endpoints. The Observatory lazily renders the 986-neuron topology in WebGL, fetches actual activations and checkpoint weights, and visualizes signed sampled contributions during staged signal playback.

## Requirements

- Node.js 22.11 or newer (the frontend and Python backend use separate environments).
- The backend running at `http://127.0.0.1:8000` with a trained MNIST checkpoint for predictions.

## Windows development

```powershell
Copy-Item .env.example .env.local
npm ci
npm run dev
```

Open `http://localhost:3000` (or `http://127.0.0.1:3000`). If the Python API uses another URL, update `NEXT_PUBLIC_API_BASE_URL` in `.env.local`. The backend allows both local Next.js origins by default.

Run quality checks from this directory:

```powershell
npm run lint
npm run typecheck
npm test
npm run build
```

Digit Studio provides pointer/touch and keyboard drawing, brush sizing, erase, undo/redo, clear, image upload, MNIST-style preprocessing preview, real backend predictions, top-five probabilities, inference time, and in-memory session history. The Neural Observatory renders all 986 neurons with instanced geometry, a sampled and batched set of weighted connections, layer-focus controls, orbit/zoom/reset camera controls, contribution details, and signal playback. The latest processed 28×28 input is stored in tab-scoped `sessionStorage` to share it between routes; the original upload is not saved by the frontend, and the backend does not persist submitted images.

The neuron inspector supports 3D selection and keyboard-accessible layer/index selectors. It shows every incoming connection for the selected neuron, including connections not rendered in the 3D sample. For hidden-layer-2 neurons it also shows direct contributions to output logits; these are not changes in softmax probabilities. Neuron and pixel indices are zero-based.

The `/training` page displays actual bounded local training progress, epoch curves, gradients, a parameter update, and isolated checkpoint information. The backend lab is disabled by default; follow the [backend setup](../backend/README.md#local-training-laboratory) to enable it locally. It is not safe to enable on a public or proxied deployment.

The `/explorer` page compares ReLU, Leaky ReLU (slope 0.01), sigmoid, tanh, and the tanh approximation of GELU, including derivatives and a hypothetical neuron sandbox. The separate parameter explorer reads the loaded inference checkpoint's actual weights and biases, showing per-layer histograms, sign counts, and strongest/weakest weights by magnitude. Sandbox changes never alter a model.

The `/analytics` page shows the checkpoint-matched official test report, confusion matrix, per-class metrics and filterable error examples. Select an error or use the current tab's Digit Studio input to view all class probabilities, backend inference latency and a target-selectable signed pixel gradient. Sensitivity is a local approximation, not an explanation of reasoning. Model comparison is not yet implemented; it would require a second model evaluated on the same test data. Follow the [analytics report instructions](../backend/README.md#analytics-report) to generate the report locally.
