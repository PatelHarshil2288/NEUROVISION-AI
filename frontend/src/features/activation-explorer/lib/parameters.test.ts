import { describe, expect, it } from "vitest";
import { rankConnections, summarize } from "./parameters";

describe("model parameter summaries", () => {
  it("counts signs and puts boundaries and the maximum in histogram bins exactly once", () => {
    const result = summarize([-2, -1, 0, 1, 2], 4);
    expect([result.negative, result.zero, result.positive]).toEqual([2, 1, 2]);
    expect(result.bins.map((bin) => bin.count)).toEqual([1, 1, 1, 2]);
    expect(result.mean).toBe(0);
  });

  it("handles identical values and rejects invalid data", () => {
    expect(summarize([0, 0, 0]).bins).toEqual([{ lower: 0, upper: 0, count: 3 }]);
    expect(() => summarize([])).toThrow();
    expect(() => summarize([NaN])).toThrow();
  });

  it("ranks actual connection weights by magnitude without losing indices", () => {
    const result = rankConnections({ weights: [[-2, 0.1], [1, -0.01]], biases: [0, 0] }, 2);
    expect(result.strongest).toEqual([
      { sourceIndex: 0, targetIndex: 0, value: -2 },
      { sourceIndex: 0, targetIndex: 1, value: 1 },
    ]);
    expect(result.weakest).toEqual([
      { sourceIndex: 1, targetIndex: 1, value: -0.01 },
      { sourceIndex: 1, targetIndex: 0, value: 0.1 },
    ]);
    expect(() => rankConnections({ weights: [], biases: [] }, 0)).toThrow();
  });
});
