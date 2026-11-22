"use client";

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from "react";

import {
  EmptyImageError,
  preprocessImageData,
} from "../lib/preprocess";

const CANVAS_SIZE = 280;
const MODEL_SIZE = 28;
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const MAX_UPLOAD_DIMENSION = 4096;
const CANVAS_BACKGROUND = "#050b13";
const INK_COLOR = "#f4f7ff";
const MAX_HISTORY = 30;

export type DrawingTool = "draw" | "erase";

export interface HistoryAvailability {
  canUndo: boolean;
  canRedo: boolean;
}

export interface DrawingCanvasHandle {
  undo: () => void;
  redo: () => void;
  clear: () => void;
  loadUpload: (file: File) => Promise<void>;
}

interface DrawingCanvasProps {
  tool: DrawingTool;
  brushSize: number;
  onPixelsChange: (pixels: number[][] | null) => void;
  onHistoryChange: (availability: HistoryAvailability) => void;
}

interface Point {
  x: number;
  y: number;
}

const DrawingCanvas = forwardRef<DrawingCanvasHandle, DrawingCanvasProps>(
  function DrawingCanvas(
    { tool, brushSize, onPixelsChange, onHistoryChange },
    ref,
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const undoStack = useRef<ImageData[]>([]);
    const redoStack = useRef<ImageData[]>([]);
    const activePointer = useRef<number | null>(null);
    const lastPoint = useRef<Point | null>(null);
    const cursorRef = useRef<Point>({ x: CANVAS_SIZE / 2, y: CANVAS_SIZE / 2 });
    const [cursor, setCursor] = useState(cursorRef.current);
    const [isFocused, setIsFocused] = useState(false);

    function getContext(): CanvasRenderingContext2D | null {
      return canvasRef.current?.getContext("2d", {
        willReadFrequently: true,
      }) ?? null;
    }

    function publishHistory(): void {
      onHistoryChange({
        canUndo: undoStack.current.length > 0,
        canRedo: redoStack.current.length > 0,
      });
    }

    function publishPixels(): void {
      const canvas = canvasRef.current;
      const context = getContext();
      if (!canvas || !context) {
        onPixelsChange(null);
        return;
      }
      try {
        onPixelsChange(
          preprocessImageData(
            context.getImageData(0, 0, canvas.width, canvas.height),
          ),
        );
      } catch (error) {
        if (error instanceof EmptyImageError) {
          onPixelsChange(null);
          return;
        }
        throw error;
      }
    }

    function saveSnapshot(): boolean {
      const canvas = canvasRef.current;
      const context = getContext();
      if (!canvas || !context) {
        return false;
      }
      undoStack.current = [
        ...undoStack.current,
        context.getImageData(0, 0, canvas.width, canvas.height),
      ].slice(-MAX_HISTORY);
      redoStack.current = [];
      publishHistory();
      return true;
    }

    function paintDot(point: Point, activeTool: DrawingTool): void {
      const context = getContext();
      if (!context) {
        return;
      }
      context.beginPath();
      context.arc(point.x, point.y, brushSize / 2, 0, Math.PI * 2);
      context.fillStyle =
        activeTool === "erase" ? CANVAS_BACKGROUND : INK_COLOR;
      context.fill();
      context.closePath();
    }

    function readPoint(
      event: PointerEvent<HTMLCanvasElement>,
    ): Point | null {
      const canvas = canvasRef.current;
      const bounds = canvas?.getBoundingClientRect();
      if (!canvas || !bounds || bounds.width <= 0 || bounds.height <= 0) {
        return null;
      }
      return {
        x: Math.max(
          0,
          Math.min(
            canvas.width,
            ((event.clientX - bounds.left) / bounds.width) * canvas.width,
          ),
        ),
        y: Math.max(
          0,
          Math.min(
            canvas.height,
            ((event.clientY - bounds.top) / bounds.height) * canvas.height,
          ),
        ),
      };
    }

    function handlePointerDown(event: PointerEvent<HTMLCanvasElement>): void {
      if (event.button !== 0 || !saveSnapshot()) {
        return;
      }
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      const point = readPoint(event);
      if (!point) {
        return;
      }
      activePointer.current = event.pointerId;
      lastPoint.current = point;
      cursorRef.current = point;
      setCursor(point);
      paintDot(point, tool);
    }

    function handlePointerMove(event: PointerEvent<HTMLCanvasElement>): void {
      if (activePointer.current !== event.pointerId) {
        return;
      }
      const point = readPoint(event);
      const previous = lastPoint.current;
      const context = getContext();
      if (!point || !previous || !context) {
        return;
      }
      context.beginPath();
      context.moveTo(previous.x, previous.y);
      context.lineTo(point.x, point.y);
      context.lineWidth = brushSize;
      context.lineCap = "round";
      context.lineJoin = "round";
      context.strokeStyle =
        tool === "erase" ? CANVAS_BACKGROUND : INK_COLOR;
      context.stroke();
      context.closePath();
      lastPoint.current = point;
      cursorRef.current = point;
      setCursor(point);
    }

    function finishPointer(event: PointerEvent<HTMLCanvasElement>): void {
      if (activePointer.current !== event.pointerId) {
        return;
      }
      activePointer.current = null;
      lastPoint.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      publishPixels();
    }

    function undo(): void {
      const canvas = canvasRef.current;
      const context = getContext();
      const previous = undoStack.current.pop();
      if (!canvas || !context || !previous) {
        return;
      }
      redoStack.current.push(
        context.getImageData(0, 0, canvas.width, canvas.height),
      );
      context.putImageData(previous, 0, 0);
      publishHistory();
      publishPixels();
    }

    function redo(): void {
      const canvas = canvasRef.current;
      const context = getContext();
      const next = redoStack.current.pop();
      if (!canvas || !context || !next) {
        return;
      }
      undoStack.current.push(
        context.getImageData(0, 0, canvas.width, canvas.height),
      );
      context.putImageData(next, 0, 0);
      publishHistory();
      publishPixels();
    }

    function clear(): void {
      const canvas = canvasRef.current;
      const context = getContext();
      if (!canvas || !context || !saveSnapshot()) {
        return;
      }
      context.fillStyle = CANVAS_BACKGROUND;
      context.fillRect(0, 0, canvas.width, canvas.height);
      publishPixels();
    }

    function handleKeyDown(event: KeyboardEvent<HTMLCanvasElement>): void {
      const movement = Math.max(4, brushSize);
      const next = { ...cursorRef.current };
      if (event.key === "ArrowLeft") {
        next.x = Math.max(0, next.x - movement);
      } else if (event.key === "ArrowRight") {
        next.x = Math.min(CANVAS_SIZE, next.x + movement);
      } else if (event.key === "ArrowUp") {
        next.y = Math.max(0, next.y - movement);
      } else if (event.key === "ArrowDown") {
        next.y = Math.min(CANVAS_SIZE, next.y + movement);
      } else if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        if (saveSnapshot()) {
          paintDot(cursorRef.current, tool);
          publishPixels();
        }
        return;
      } else {
        return;
      }
      event.preventDefault();
      cursorRef.current = next;
      setCursor(next);
    }

    async function loadUpload(file: File): Promise<void> {
      if (!file.type.startsWith("image/")) {
        throw new Error("Choose a PNG, JPEG, WebP, or other image file.");
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        throw new Error("Image uploads must be 10 MB or smaller.");
      }

      const bitmap = await createImageBitmap(file);
      try {
        if (
          bitmap.width > MAX_UPLOAD_DIMENSION ||
          bitmap.height > MAX_UPLOAD_DIMENSION
        ) {
          throw new Error("Image dimensions must be 4096 pixels or smaller.");
        }
        const scale = Math.min(
          1,
          2048 / Math.max(bitmap.width, bitmap.height),
        );
        const source = document.createElement("canvas");
        source.width = Math.max(1, Math.round(bitmap.width * scale));
        source.height = Math.max(1, Math.round(bitmap.height * scale));
        const sourceContext = source.getContext("2d", {
          willReadFrequently: true,
        });
        if (!sourceContext) {
          throw new Error("This browser cannot process the selected image.");
        }
        sourceContext.drawImage(bitmap, 0, 0, source.width, source.height);
        const pixels = preprocessImageData(
          sourceContext.getImageData(0, 0, source.width, source.height),
        );

        const canvas = canvasRef.current;
        const context = getContext();
        if (!canvas || !context) {
          throw new Error("The drawing canvas is unavailable.");
        }
        if (!saveSnapshot()) {
          throw new Error("The drawing canvas is unavailable.");
        }

        const modelImage = document.createElement("canvas");
        modelImage.width = MODEL_SIZE;
        modelImage.height = MODEL_SIZE;
        const modelContext = modelImage.getContext("2d");
        if (!modelContext) {
          throw new Error("This browser cannot render the processed image.");
        }
        const imageData = modelContext.createImageData(
          MODEL_SIZE,
          MODEL_SIZE,
        );
        for (let y = 0; y < MODEL_SIZE; y += 1) {
          for (let x = 0; x < MODEL_SIZE; x += 1) {
            const offset = (y * MODEL_SIZE + x) * 4;
            const intensity = Math.round(pixels[y][x] * 255);
            imageData.data[offset] = intensity;
            imageData.data[offset + 1] = intensity;
            imageData.data[offset + 2] = intensity;
            imageData.data[offset + 3] = 255;
          }
        }
        modelContext.putImageData(imageData, 0, 0);
        context.fillStyle = CANVAS_BACKGROUND;
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.imageSmoothingEnabled = false;
        context.drawImage(modelImage, 0, 0, canvas.width, canvas.height);
        publishHistory();
        onPixelsChange(pixels);
      } finally {
        bitmap.close();
      }
    }

    useImperativeHandle(
      ref,
      () => ({
        undo,
        redo,
        clear,
        loadUpload,
      }),
    );

    useEffect(() => {
      const canvas = canvasRef.current;
      const context = getContext();
      if (!canvas || !context) {
        return;
      }
      context.fillStyle = CANVAS_BACKGROUND;
      context.fillRect(0, 0, canvas.width, canvas.height);
      onHistoryChange({
        canUndo: undoStack.current.length > 0,
        canRedo: redoStack.current.length > 0,
      });
    }, [onHistoryChange]);

    return (
      <div className="drawing-surface">
        <canvas
          ref={canvasRef}
          aria-label="Drawing canvas. Use a pointer or arrow keys to position the cursor, then press Enter to draw."
          className="drawing-canvas"
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          tabIndex={0}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          onKeyDown={handleKeyDown}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={finishPointer}
          onPointerCancel={finishPointer}
        />
        {isFocused && (
          <span
            aria-hidden="true"
            className="keyboard-cursor"
            style={{
              left: `${(cursor.x / CANVAS_SIZE) * 100}%`,
              top: `${(cursor.y / CANVAS_SIZE) * 100}%`,
            }}
          />
        )}
      </div>
    );
  },
);

export default DrawingCanvas;
