export type LayerId = "input" | "hidden_1" | "hidden_2" | "output";

export interface NetworkLayer {
  id: LayerId;
  label: string;
  shortLabel: string;
  neuronCount: number;
  columns: number;
  rows: number;
  positionZ: number;
  spacing: number;
  radius: number;
  color: string;
  sampledConnectionsPerNeuron: number;
}

export interface NetworkNeuron {
  layerId: LayerId;
  index: number;
  position: [number, number, number];
  radius: number;
  color: string;
}

export interface NetworkConnection {
  from: number;
  to: number;
  sourceIndex: number;
  targetIndex: number;
  transition: number;
}

export interface NetworkTopology {
  neurons: NetworkNeuron[];
  connections: NetworkConnection[];
  totalNeuronCount: number;
  totalConnectionCount: number;
}

export const NETWORK_LAYERS: readonly NetworkLayer[] = [
  {
    id: "input",
    label: "Input layer",
    shortLabel: "INPUT",
    neuronCount: 784,
    columns: 28,
    rows: 28,
    positionZ: -9,
    spacing: 0.18,
    radius: 0.055,
    color: "#66e4e6",
    sampledConnectionsPerNeuron: 12,
  },
  {
    id: "hidden_1",
    label: "Hidden layer 1",
    shortLabel: "HIDDEN 01",
    neuronCount: 128,
    columns: 8,
    rows: 16,
    positionZ: -3,
    spacing: 0.27,
    radius: 0.09,
    color: "#9c91ff",
    sampledConnectionsPerNeuron: 8,
  },
  {
    id: "hidden_2",
    label: "Hidden layer 2",
    shortLabel: "HIDDEN 02",
    neuronCount: 64,
    columns: 8,
    rows: 8,
    positionZ: 3,
    spacing: 0.34,
    radius: 0.11,
    color: "#ba9dff",
    sampledConnectionsPerNeuron: 8,
  },
  {
    id: "output",
    label: "Output layer",
    shortLabel: "OUTPUT",
    neuronCount: 10,
    columns: 1,
    rows: 10,
    positionZ: 9,
    spacing: 0.58,
    radius: 0.17,
    color: "#f0c77e",
    sampledConnectionsPerNeuron: 0,
  },
];

export const TOTAL_CONNECTION_COUNT =
  784 * 128 + 128 * 64 + 64 * 10;

export function createNetworkTopology(): NetworkTopology {
  const neurons: NetworkNeuron[] = [];
  const layerOffsets = new Map<LayerId, number>();

  for (const layer of NETWORK_LAYERS) {
    layerOffsets.set(layer.id, neurons.length);
    for (let index = 0; index < layer.neuronCount; index += 1) {
      const column = index % layer.columns;
      const row = Math.floor(index / layer.columns);
      neurons.push({
        layerId: layer.id,
        index,
        position: [
          (column - (layer.columns - 1) / 2) * layer.spacing,
          ((layer.rows - 1) / 2 - row) * layer.spacing,
          layer.positionZ,
        ],
        radius: layer.radius,
        color: layer.color,
      });
    }
  }

  const connections: NetworkConnection[] = [];
  for (let transition = 0; transition < NETWORK_LAYERS.length - 1; transition += 1) {
    const sourceLayer = NETWORK_LAYERS[transition];
    const targetLayer = NETWORK_LAYERS[transition + 1];
    const sourceOffset = layerOffsets.get(sourceLayer.id);
    const targetOffset = layerOffsets.get(targetLayer.id);
    if (sourceOffset === undefined || targetOffset === undefined) {
      throw new Error("Every network layer must have a neuron offset.");
    }

    for (let targetIndex = 0; targetIndex < targetLayer.neuronCount; targetIndex += 1) {
      for (
        let sampleIndex = 0;
        sampleIndex < sourceLayer.sampledConnectionsPerNeuron;
        sampleIndex += 1
      ) {
        const sourceIndex =
          (targetIndex * 67 + sampleIndex * 113) % sourceLayer.neuronCount;
        connections.push({
          from: sourceOffset + sourceIndex,
          to: targetOffset + targetIndex,
          sourceIndex,
          targetIndex,
          transition,
        });
      }
    }
  }

  return {
    neurons,
    connections,
    totalNeuronCount: neurons.length,
    totalConnectionCount: TOTAL_CONNECTION_COUNT,
  };
}

export const NETWORK_TOPOLOGY = createNetworkTopology();
