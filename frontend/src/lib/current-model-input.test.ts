import { afterEach, describe, expect, it } from "vitest";

import {
  clearCurrentModelInput,
  isModelInputPixels,
  readCurrentModelInput,
  saveCurrentModelInput,
} from "./current-model-input";

afterEach(() => {
  window.sessionStorage.clear();
});

describe("current model input session storage", () => {
  it("persists and reads a finite 28 by 28 grayscale image", () => {
    const pixels = Array.from({ length: 28 }, (_, row) =>
      Array.from({ length: 28 }, (_, column) => (row + column) / 54),
    );

    saveCurrentModelInput(pixels);

    expect(readCurrentModelInput()).toEqual(pixels);
  });

  it("rejects malformed or out-of-range model inputs", () => {
    expect(isModelInputPixels([[0]])).toBe(false);
    expect(
      isModelInputPixels(
        Array.from({ length: 28 }, () => Array<number>(28).fill(Number.NaN)),
      ),
    ).toBe(false);
    expect(() => saveCurrentModelInput([[0]])).toThrow("finite 28×28 image");
  });

  it("clears the current input and reports corrupt saved data", () => {
    window.sessionStorage.setItem(
      "neurovision.current-model-input.v1",
      "not-json",
    );

    expect(() => readCurrentModelInput()).toThrow("Saved image data is corrupted");
    expect(window.sessionStorage.length).toBe(0);

    saveCurrentModelInput(
      Array.from({ length: 28 }, () => Array<number>(28).fill(0.25)),
    );
    clearCurrentModelInput();
    expect(readCurrentModelInput()).toBeNull();
  });
});
