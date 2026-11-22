const STORAGE_KEY = "neurovision.current-model-input.v1";
const IMAGE_SIZE = 28;

export type ModelInputPixels = number[][];

export function isModelInputPixels(value: unknown): value is ModelInputPixels {
  return (
    Array.isArray(value) &&
    value.length === IMAGE_SIZE &&
    value.every(
      (row) =>
        Array.isArray(row) &&
        row.length === IMAGE_SIZE &&
        row.every(
          (pixel) =>
            typeof pixel === "number" &&
            Number.isFinite(pixel) &&
            pixel >= 0 &&
            pixel <= 1,
        ),
    )
  );
}

export function saveCurrentModelInput(pixels: ModelInputPixels): void {
  if (!isModelInputPixels(pixels)) {
    throw new Error("The current model input must be a finite 28×28 image.");
  }

  const storage = getSessionStorage();
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(pixels));
  } catch {
    throw new Error(
      "The current image could not be shared with the Neural Observatory.",
    );
  }
}

export function readCurrentModelInput(): ModelInputPixels | null {
  const storage = getSessionStorage();
  let serialized: string | null;
  try {
    serialized = storage.getItem(STORAGE_KEY);
  } catch {
    throw new Error(
      "The current image could not be read from this browser session.",
    );
  }

  if (serialized === null) {
    return null;
  }

  let value: unknown;
  try {
    value = JSON.parse(serialized);
  } catch {
    removeInvalidInput(storage);
    throw new Error("Saved image data is corrupted. Draw or upload the digit again.");
  }

  if (!isModelInputPixels(value)) {
    removeInvalidInput(storage);
    throw new Error(
      "Saved image data is invalid. Draw or upload the digit again.",
    );
  }
  return value;
}

export function clearCurrentModelInput(): void {
  const storage = getSessionStorage();
  try {
    storage.removeItem(STORAGE_KEY);
  } catch {
    throw new Error(
      "The current image could not be cleared from this browser session.",
    );
  }
}

function getSessionStorage(): Storage {
  if (typeof window === "undefined") {
    throw new Error("This image can only be shared from an active browser tab.");
  }

  try {
    return window.sessionStorage;
  } catch {
    throw new Error(
      "Browser session storage is unavailable, so the image cannot be shared.",
    );
  }
}

function removeInvalidInput(storage: Storage): void {
  try {
    storage.removeItem(STORAGE_KEY);
  } catch {
    throw new Error(
      "Invalid saved image data could not be cleared from this browser session.",
    );
  }
}
