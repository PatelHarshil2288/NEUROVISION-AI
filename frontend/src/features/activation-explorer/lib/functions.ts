export type ActivationId = "relu" | "leaky-relu" | "sigmoid" | "tanh" | "gelu";

export interface ActivationDefinition {
  id: ActivationId;
  name: string;
  equation: string;
  derivativeEquation: string;
  advantage: string;
  limitation: string;
}

export const ACTIVATIONS: readonly ActivationDefinition[] = [
  {
    id: "relu",
    name: "ReLU",
    equation: "f(x) = max(0, x)",
    derivativeEquation: "f′(x) = 0 for x < 0; 1 for x > 0; undefined at 0",
    advantage: "Simple and computationally inexpensive.",
    limitation: "Negative inputs have zero gradient, so units can stop learning.",
  },
  {
    id: "leaky-relu",
    name: "Leaky ReLU",
    equation: "f(x) = x if x ≥ 0, otherwise 0.01x",
    derivativeEquation: "f′(x) = 0.01 for x < 0; 1 for x > 0; undefined at 0",
    advantage: "Retains a small gradient for negative inputs.",
    limitation: "The negative-side slope is a design choice and may not suit every problem.",
  },
  {
    id: "sigmoid",
    name: "Sigmoid",
    equation: "f(x) = 1 / (1 + e⁻ˣ)",
    derivativeEquation: "f′(x) = f(x)(1 − f(x))",
    advantage: "Produces a smooth value between 0 and 1.",
    limitation: "Gradients become small for inputs far from zero.",
  },
  {
    id: "tanh",
    name: "Tanh",
    equation: "f(x) = tanh(x)",
    derivativeEquation: "f′(x) = 1 − tanh²(x)",
    advantage: "Smooth, zero-centered output between −1 and 1.",
    limitation: "Gradients become small when the function saturates.",
  },
  {
    id: "gelu",
    name: "GELU (tanh approximation)",
    equation: "f(x) ≈ ½x[1 + tanh(√(2/π)(x + 0.044715x³))]",
    derivativeEquation: "f′(x) ≈ ½(1 + tanh(u)) + ½x(1 − tanh²(u))√(2/π)(1 + 0.134145x²), where u = √(2/π)(x + 0.044715x³)",
    advantage: "Smoothly scales inputs rather than applying a hard cutoff.",
    limitation: "More expensive than ReLU; this explorer uses the common tanh approximation, not exact GELU.",
  },
];

export function activation(id: ActivationId, x: number): number {
  switch (id) {
    case "relu": return Math.max(0, x);
    case "leaky-relu": return x >= 0 ? x : 0.01 * x;
    case "sigmoid": return 1 / (1 + Math.exp(-x));
    case "tanh": return Math.tanh(x);
    case "gelu": {
      const u = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3);
      return 0.5 * x * (1 + Math.tanh(u));
    }
  }
}

export function derivative(id: ActivationId, x: number): number | null {
  switch (id) {
    case "relu": return x === 0 ? null : x > 0 ? 1 : 0;
    case "leaky-relu": return x === 0 ? null : x > 0 ? 1 : 0.01;
    case "sigmoid": {
      const y = activation(id, x);
      return y * (1 - y);
    }
    case "tanh": return 1 - Math.tanh(x) ** 2;
    case "gelu": {
      const c = Math.sqrt(2 / Math.PI);
      const u = c * (x + 0.044715 * x ** 3);
      const t = Math.tanh(u);
      return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 0.134145 * x * x);
    }
  }
}
