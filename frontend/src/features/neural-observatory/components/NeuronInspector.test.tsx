import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it } from "vitest";

import type { ActivationsResult, ModelWeightsResult } from "../../digit-studio/lib/api";
import type { NeuronSelection } from "../lib/signal";
import NeuronInspector from "./NeuronInspector";

function TestInspector() {
  const [selection, setSelection] = useState<NeuronSelection>({ layerId: "input", index: 0 });
  const activations: ActivationsResult = {
    model_id: "mnist-mlp-v1", model_version: "v1", predicted_class: 0,
    logits: Array(10).fill(0), probabilities: Array(10).fill(0.1),
    layers: {
      input: { pre_activations: [], activations: Array(784).fill(0) },
      hidden_1: { pre_activations: Array(128).fill(0), activations: Array(128).fill(0) },
      hidden_2: { pre_activations: Array(64).fill(0), activations: Array(64).fill(0) },
      output: { pre_activations: Array(10).fill(0), activations: Array(10).fill(0.1) },
    },
  };
  const layer = (inputs: number, outputs: number) => ({
    weights: Array.from({ length: outputs }, () => Array(inputs).fill(0)),
    biases: Array(outputs).fill(0),
  });
  const weights: ModelWeightsResult = {
    model_id: "mnist-mlp-v1", model_version: "v1", architecture: [784, 128, 64, 10],
    layers: {
      input_to_hidden_1: layer(784, 128),
      hidden_1_to_hidden_2: layer(128, 64),
      hidden_2_to_output: layer(64, 10),
    },
  };
  return <NeuronInspector selection={selection} onSelect={setSelection} activations={activations} weights={weights} pixels={Array.from({ length: 28 }, () => Array(28).fill(0))} />;
}

afterEach(cleanup);

describe("NeuronInspector", () => {
  it("supports selecting neurons and explains input, hidden, and output values", () => {
    render(<TestInspector />);
    fireEvent.change(screen.getByLabelText("Neuron index"), { target: { value: "29" } });
    expect(screen.getByText(/Pixel location: row 1, column 1/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Inspector layer"), { target: { value: "hidden_1" } });
    expect(screen.getByRole("heading", { name: "Hidden layer 1 · neuron 0" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Incoming connections (784)" })).toBeInTheDocument();
    expect(within(screen.getByRole("table")).getAllByRole("row")).toHaveLength(785);

    fireEvent.change(screen.getByLabelText("Inspector layer"), { target: { value: "output" } });
    expect(screen.getByRole("heading", { name: "Incoming connections (64)" })).toBeInTheDocument();
    expect(screen.getByText(/Softmax converts all 10 logits/)).toBeInTheDocument();
  });
});
