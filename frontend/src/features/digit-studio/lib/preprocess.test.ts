import { describe, expect, it } from "vitest";

import {
  EmptyImageError,
  preprocessImageData,
  type ImageFrame,
} from "./preprocess";

function makeDigitFrame(
  background: number,
  foreground: number,
): ImageFrame {
  const width = 48;
  const height = 40;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    data[offset] = background;
    data[offset + 1] = background;
    data[offset + 2] = background;
    data[offset + 3] = 255;
  }
  for (let y = 8; y < 32; y += 1) {
    for (let x = 5; x < 11; x += 1) {
      const offset = (y * width + x) * 4;
      data[offset] = foreground;
      data[offset + 1] = foreground;
      data[offset + 2] = foreground;
    }
  }
  return { width, height, data };
}

describe("preprocessImageData", () => {
  it("crops, centers, and scales a white digit on a dark background", () => {
    const pixels = preprocessImageData(makeDigitFrame(0, 255));
    const activePixels = pixels.flatMap((row, y) =>
      row.flatMap((value, x) => (value > 0 ? [{ x, y }] : [])),
    );
    const centerX =
      activePixels.reduce((sum, pixel) => sum + pixel.x, 0) /
      activePixels.length;
    const centerY =
      activePixels.reduce((sum, pixel) => sum + pixel.y, 0) /
      activePixels.length;

    expect(pixels).toHaveLength(28);
    expect(pixels.every((row) => row.length === 28)).toBe(true);
    expect(activePixels.length).toBeGreaterThan(0);
    expect(centerX).toBeCloseTo(13, 0);
    expect(centerY).toBeCloseTo(13.5, 0);
  });

  it("auto-inverts a dark digit on a light uploaded image", () => {
    const darkOnLight = preprocessImageData(makeDigitFrame(255, 0));
    const lightOnDark = preprocessImageData(makeDigitFrame(0, 255));

    expect(darkOnLight).toEqual(lightOnDark);
  });

  it("rejects empty images instead of sending blank input", () => {
    const frame = makeDigitFrame(0, 0);

    expect(() => preprocessImageData(frame)).toThrow(EmptyImageError);
  });

  it("rejects malformed image data", () => {
    expect(() =>
      preprocessImageData({
        width: 28,
        height: 28,
        data: new Uint8ClampedArray(4),
      }),
    ).toThrow("Image data has invalid dimensions.");
  });
});
