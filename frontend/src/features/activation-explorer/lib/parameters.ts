import type { LayerWeightsData } from "../../digit-studio/lib/api";

export interface HistogramBin {
  lower: number;
  upper: number;
  count: number;
}

export function summarize(values: number[], binCount = 20) {
  if (values.length === 0 || !Number.isInteger(binCount) || binCount < 1 ||
      values.some((value) => !Number.isFinite(value))) {
    throw new Error("Parameter values must be finite and nonempty, with a positive bin count.");
  }
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let positive = 0;
  let negative = 0;
  let zero = 0;
  for (const value of values) {
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    if (value > 0) positive += 1;
    else if (value < 0) negative += 1;
    else zero += 1;
  }
  const width = (max - min) / binCount;
  if (width === 0) {
    return { count: values.length, min, max, mean: sum / values.length, positive, negative, zero, bins: [{ lower: min, upper: max, count: values.length }] };
  }
  const bins: HistogramBin[] = Array.from({ length: binCount }, (_, index) => ({
    lower: min + width * index,
    upper: index === binCount - 1 ? max : min + width * (index + 1),
    count: 0,
  }));
  for (const value of values) {
    const index = Math.min(binCount - 1, Math.floor((value - min) / width));
    bins[index]!.count += 1;
  }
  return { count: values.length, min, max, mean: sum / values.length, positive, negative, zero, bins };
}

export interface RankedWeight {
  sourceIndex: number;
  targetIndex: number;
  value: number;
}

export function rankConnections(layer: LayerWeightsData, limit = 5) {
  if (!Number.isInteger(limit) || limit < 1) {
    throw new Error("The connection ranking limit must be a positive integer.");
  }
  const connections: RankedWeight[] = layer.weights.flatMap((row, targetIndex) =>
    row.map((value, sourceIndex) => ({ sourceIndex, targetIndex, value })),
  );
  connections.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return {
    strongest: connections.slice(0, limit),
    weakest: connections.slice(-limit).reverse(),
  };
}
