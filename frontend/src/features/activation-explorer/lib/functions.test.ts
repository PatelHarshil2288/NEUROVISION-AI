import { describe, expect, it } from "vitest";

import { ACTIVATIONS, activation, derivative } from "./functions";

describe("activation functions", () => {
  it("calculates known values and identifies nondifferentiable points", () => {
    expect(activation("relu", -2)).toBe(0);
    expect(activation("leaky-relu", -2)).toBeCloseTo(-0.02);
    expect(activation("sigmoid", 0)).toBe(0.5);
    expect(activation("tanh", 0)).toBe(0);
    expect(activation("gelu", 0)).toBe(0);
    expect(derivative("relu", 0)).toBeNull();
    expect(derivative("leaky-relu", 0)).toBeNull();
    expect(derivative("gelu", 0)).toBeCloseTo(0.5);
  });

  it("derivatives agree with numerical slopes on both sides of zero", () => {
    for (const { id } of ACTIVATIONS) {
      for (const x of [-2, -0.5, 0.5, 2]) {
        const step = 0.00001;
        const numerical = (activation(id, x + step) - activation(id, x - step)) / (2 * step);
        expect(derivative(id, x)).toBeCloseTo(numerical, 5);
      }
    }
  });
});
