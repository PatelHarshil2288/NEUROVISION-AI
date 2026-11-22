import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { getModelWeights, type ModelWeightsResult } from "../digit-studio/lib/api";
import ActivationExplorer from "./ActivationExplorer";

vi.mock("../digit-studio/lib/api", () => ({ getModelWeights: vi.fn() }));

function realWeights(): ModelWeightsResult {
  const layer = (inputs: number, outputs: number) => ({
    weights: Array.from({ length: outputs }, () => Array<number>(inputs).fill(0)),
    biases: Array<number>(outputs).fill(0),
  });
  const first = layer(784, 128);
  first.weights[1]![3] = -2;
  first.biases[0] = 0.25;
  return {
    model_id: "mnist-mlp-v1", model_version: "test-checkpoint", architecture: [784, 128, 64, 10],
    layers: { input_to_hidden_1: first, hidden_1_to_hidden_2: layer(128, 64), hidden_2_to_output: layer(64, 10) },
  };
}

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe("ActivationExplorer", () => {
  it("keeps hypothetical controls separate from real checkpoint parameters", async () => {
    vi.mocked(getModelWeights).mockResolvedValue(realWeights());
    render(<ActivationExplorer />);
    expect(await screen.findByText("test-checkpoint")).toBeInTheDocument();
    const parameters = screen.getByRole("region", { name: "Trained model parameters" });
    expect(within(parameters).getByText("100,352")).toBeInTheDocument();
    expect(within(parameters).getByText(/Source 3 → target 1/)).toHaveTextContent("-2.000000");

    fireEvent.change(screen.getByLabelText("Activation function"), { target: { value: "sigmoid" } });
    fireEvent.change(screen.getByLabelText(/Example input x/), { target: { value: "0" } });
    expect(screen.getByText(/f\(0.00\) =/)).toHaveTextContent("0.50000");
    fireEvent.change(screen.getByLabelText(/Bias b/), { target: { value: "1" } });
    expect(screen.getByText(/z = x1/)).toHaveTextContent("2.00000");
    expect(within(parameters).getByText("100,352")).toBeInTheDocument();
    expect(getModelWeights).toHaveBeenCalledTimes(1);

    fireEvent.change(screen.getByLabelText("Parameter type"), { target: { value: "biases" } });
    expect(within(parameters).getByText("128")).toBeInTheDocument();
    expect(within(parameters).queryByText("Strongest connections")).not.toBeInTheDocument();
  });

  it("reports unavailable model data without showing invented parameters", async () => {
    vi.mocked(getModelWeights).mockRejectedValue(new Error("No checkpoint loaded"));
    render(<ActivationExplorer />);
    expect(await screen.findByRole("alert")).toHaveTextContent("No checkpoint loaded");
    expect(screen.queryByText("Strongest connections")).not.toBeInTheDocument();
    expect(screen.getByText("Weight and bias sandbox")).toBeInTheDocument();
  });
});
