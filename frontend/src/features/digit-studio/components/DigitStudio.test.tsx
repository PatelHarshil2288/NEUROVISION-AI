import {
  cleanup,
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import DigitStudio from "./DigitStudio";
import {
  readCurrentModelInput,
  saveCurrentModelInput,
} from "../../../lib/current-model-input";

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function makeInkImageData(): ImageData {
  const width = 280;
  const height = 280;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    data[offset] = 5;
    data[offset + 1] = 11;
    data[offset + 2] = 19;
    data[offset + 3] = 255;
  }
  for (let y = 70; y < 210; y += 1) {
    for (let x = 120; x < 150; x += 1) {
      const offset = (y * width + x) * 4;
      data[offset] = 244;
      data[offset + 1] = 247;
      data[offset + 2] = 255;
    }
  }
  return { width, height, data } as ImageData;
}

const inkImageData = makeInkImageData();
const context = {
  fillStyle: "",
  strokeStyle: "",
  lineWidth: 0,
  lineCap: "round",
  lineJoin: "round",
  imageSmoothingEnabled: false,
  fillRect: vi.fn(),
  beginPath: vi.fn(),
  arc: vi.fn(),
  fill: vi.fn(),
  closePath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  stroke: vi.fn(),
  getImageData: vi.fn(() => inkImageData),
  putImageData: vi.fn(),
  createImageData: vi.fn(
    (width: number, height: number) =>
      ({
        width,
        height,
        data: new Uint8ClampedArray(width * height * 4),
      }) as ImageData,
  ),
  drawImage: vi.fn(),
} as unknown as CanvasRenderingContext2D;

function installCanvasMock(): void {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context);
}

function readyFetch(): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/health")) {
      return Promise.resolve(
        jsonResponse({ status: "ok", model_ready: true }),
      );
    }
    if (url.endsWith("/models")) {
      return Promise.resolve(
        jsonResponse({
          models: [
            {
              model_id: "mnist-mlp-v1",
              model_version: "mlp-mnist-v1",
              available: true,
            },
          ],
        }),
      );
    }
    return Promise.resolve(
      jsonResponse({
        model_id: "mnist-mlp-v1",
        model_version: "mlp-mnist-v1",
        predicted_class: 7,
        logits: [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
        probabilities: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.65, 0.04, 0.03],
        inference_ms: 1.25,
      }),
    );
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  cleanup();
  window.sessionStorage.clear();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("DigitStudio", () => {
  it("supports keyboard drawing, undo, and eraser selection", async () => {
    installCanvasMock();
    readyFetch();
    render(<DigitStudio />);

    const canvas = screen.getByLabelText(/Drawing canvas/);
    await screen.findByText("Model ready");
    fireEvent.keyDown(canvas, { key: "Enter" });

    expect(context.arc).toHaveBeenCalled();
    expect(readCurrentModelInput()).not.toBeNull();
    expect(screen.getByRole("button", { name: "Undo" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Erase" }));
    expect(screen.getByRole("button", { name: "Erase" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Undo" })).toBeDisabled();
    });
  });

  it("preserves the latest processed image when the empty canvas mounts", async () => {
    const pixels = Array.from({ length: 28 }, () =>
      Array<number>(28).fill(0.5),
    );
    saveCurrentModelInput(pixels);
    installCanvasMock();
    readyFetch();
    render(<DigitStudio />);

    await screen.findByText("Model ready");

    expect(readCurrentModelInput()).toEqual(pixels);
  });

  it("surfaces backend connection errors instead of fabricating a result", async () => {
    installCanvasMock();
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    render(<DigitStudio />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Cannot reach the AI service.",
    );
    expect(screen.getByText("NO PREDICTION YET")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Run prediction/ })).toBeDisabled();
  });

  it("shows predictions and records only successful API responses", async () => {
    installCanvasMock();
    readyFetch();
    render(<DigitStudio />);

    const canvas = screen.getByLabelText(/Drawing canvas/);
    await screen.findByText("Model ready");
    fireEvent.keyDown(canvas, { key: "Enter" });
    fireEvent.click(screen.getByRole("button", { name: /Run prediction/ }));

    expect(await screen.findByText("PREDICTED DIGIT")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Probability for digit 7" })).toHaveAttribute(
      "aria-valuenow",
      "65",
    );
    expect(screen.getByText("1 RUN")).toBeInTheDocument();
  });

  it("discards a prediction when the input changes before the response arrives", async () => {
    installCanvasMock();
    const fetchMock = readyFetch();
    let resolvePrediction!: (response: Response) => void;
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      if (String(input).endsWith("/predict")) {
        return new Promise<Response>((resolve) => { resolvePrediction = resolve; });
      }
      if (String(input).endsWith("/health")) {
        return Promise.resolve(jsonResponse({ status: "ok", model_ready: true }));
      }
      return Promise.resolve(jsonResponse({ models: [{ model_id: "mnist-mlp-v1", model_version: "mlp-mnist-v1", available: true }] }));
    });
    render(<DigitStudio />);
    await screen.findByText("Model ready");
    fireEvent.keyDown(screen.getByLabelText(/Drawing canvas/), { key: "Enter" });
    fireEvent.click(screen.getByRole("button", { name: /Run prediction/ }));
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    await act(async () => resolvePrediction(jsonResponse({
      model_id: "mnist-mlp-v1", model_version: "mlp-mnist-v1",
      predicted_class: 7, logits: [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
      probabilities: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.65, 0.04, 0.03],
      inference_ms: 1.25,
    })));
    expect(screen.getByText("NO PREDICTION YET")).toBeInTheDocument();
    expect(screen.queryByText("1 RUN")).not.toBeInTheDocument();
  });
});
