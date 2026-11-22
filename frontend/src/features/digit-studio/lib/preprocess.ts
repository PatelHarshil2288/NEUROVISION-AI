export interface ImageFrame {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export class EmptyImageError extends Error {
  constructor() {
    super("No visible digit was found. Draw or upload a darker foreground shape.");
    this.name = "EmptyImageError";
  }
}

const MODEL_SIZE = 28;
const DIGIT_SIZE = 20;
const INK_THRESHOLD = 24;

export function preprocessImageData(frame: ImageFrame): number[][] {
  const { width, height, data } = frame;
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    data.length !== width * height * 4
  ) {
    throw new Error("Image data has invalid dimensions.");
  }

  const grayscale = new Float32Array(width * height);
  for (let pixel = 0; pixel < grayscale.length; pixel += 1) {
    const offset = pixel * 4;
    const alpha = data[offset + 3] / 255;
    const luminance =
      data[offset] * 0.2126 +
      data[offset + 1] * 0.7152 +
      data[offset + 2] * 0.0722;
    grayscale[pixel] = luminance * alpha + 255 * (1 - alpha);
  }

  const border = sampleBorder(grayscale, width, height);
  const invert = median(border) > 127;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const intensity = foregroundIntensity(
        grayscale[y * width + x],
        invert,
      );
      if (intensity > INK_THRESHOLD) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX < minX || maxY < minY) {
    throw new EmptyImageError();
  }

  const sourceWidth = maxX - minX + 1;
  const sourceHeight = maxY - minY + 1;
  const scale = DIGIT_SIZE / Math.max(sourceWidth, sourceHeight);
  const targetWidth = Math.max(1, Math.round(sourceWidth * scale));
  const targetHeight = Math.max(1, Math.round(sourceHeight * scale));
  const offsetX = Math.floor((MODEL_SIZE - targetWidth) / 2);
  const offsetY = Math.floor((MODEL_SIZE - targetHeight) / 2);
  const pixels = Array.from({ length: MODEL_SIZE }, () =>
    Array<number>(MODEL_SIZE).fill(0),
  );

  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = minY + ((y + 0.5) * sourceHeight) / targetHeight - 0.5;
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = minX + ((x + 0.5) * sourceWidth) / targetWidth - 0.5;
      const intensity = sampleBilinear(
        grayscale,
        width,
        sourceX,
        sourceY,
        minX,
        minY,
        maxX,
        maxY,
        invert,
      );
      pixels[offsetY + y][offsetX + x] = intensity / 255;
    }
  }

  return pixels;
}

function foregroundIntensity(value: number, invert: boolean): number {
  return invert ? 255 - value : value;
}

function sampleBorder(
  grayscale: Float32Array,
  width: number,
  height: number,
): number[] {
  const samples: number[] = [];
  const step = Math.max(1, Math.floor(Math.min(width, height) / 256));

  for (let x = 0; x < width; x += step) {
    samples.push(grayscale[x], grayscale[(height - 1) * width + x]);
  }
  for (let y = step; y < height - 1; y += step) {
    samples.push(grayscale[y * width], grayscale[y * width + width - 1]);
  }
  return samples;
}

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }
  return sorted[middle];
}

function sampleBilinear(
  grayscale: Float32Array,
  width: number,
  sourceX: number,
  sourceY: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  invert: boolean,
): number {
  const x = Math.min(maxX, Math.max(minX, sourceX));
  const y = Math.min(maxY, Math.max(minY, sourceY));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(maxX, x0 + 1);
  const y1 = Math.min(maxY, y0 + 1);
  const dx = x - x0;
  const dy = y - y0;
  const top =
    grayscale[y0 * width + x0] * (1 - dx) +
    grayscale[y0 * width + x1] * dx;
  const bottom =
    grayscale[y1 * width + x0] * (1 - dx) +
    grayscale[y1 * width + x1] * dx;
  const value = top * (1 - dy) + bottom * dy;
  return foregroundIntensity(value, invert);
}
