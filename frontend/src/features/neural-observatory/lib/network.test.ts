import { describe, expect, it } from "vitest";

import {
  createNetworkTopology,
  NETWORK_LAYERS,
  NETWORK_TOPOLOGY,
  TOTAL_CONNECTION_COUNT,
} from "./network";

describe("neural network topology", () => {
  it("lays out all 986 neurons in the four model layers", () => {
    expect(NETWORK_TOPOLOGY.totalNeuronCount).toBe(986);
    expect(NETWORK_LAYERS.map((layer) => layer.neuronCount)).toEqual([
      784, 128, 64, 10,
    ]);

    for (const layer of NETWORK_LAYERS) {
      const layerNeurons = NETWORK_TOPOLOGY.neurons.filter(
        (neuron) => neuron.layerId === layer.id,
      );
      expect(layerNeurons).toHaveLength(layer.neuronCount);
      expect(layerNeurons.every((neuron) => neuron.position[2] === layer.positionZ))
        .toBe(true);
    }
  });

  it("renders a deterministic, bounded sample of fully connected edges", () => {
    expect(NETWORK_TOPOLOGY.totalConnectionCount).toBe(TOTAL_CONNECTION_COUNT);
    expect(TOTAL_CONNECTION_COUNT).toBe(109_184);
    expect(NETWORK_TOPOLOGY.connections).toHaveLength(2_128);
    expect(createNetworkTopology()).toEqual(NETWORK_TOPOLOGY);

    for (const connection of NETWORK_TOPOLOGY.connections) {
      expect(connection.from).toBeGreaterThanOrEqual(0);
      expect(connection.from).toBeLessThan(NETWORK_TOPOLOGY.neurons.length);
      expect(connection.to).toBeGreaterThan(connection.from);
      expect(connection.to).toBeLessThan(NETWORK_TOPOLOGY.neurons.length);

      const source = NETWORK_TOPOLOGY.neurons[connection.from];
      const target = NETWORK_TOPOLOGY.neurons[connection.to];
      const sourceLayer = NETWORK_LAYERS[connection.transition];
      const targetLayer = NETWORK_LAYERS[connection.transition + 1];
      expect(source?.layerId).toBe(sourceLayer?.id);
      expect(target?.layerId).toBe(targetLayer?.id);
      expect(source?.index).toBe(connection.sourceIndex);
      expect(target?.index).toBe(connection.targetIndex);
    }
  });
});
