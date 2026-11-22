import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { predictDigit } from "../digit-studio/lib/api";
import { readCurrentModelInput } from "../../lib/current-model-input";
import {
  activateModel,
  compareModels,
  evaluateCheckpoint,
  getComparisonHistory,
  getComparisonReports,
  getErrorImage,
  getEvaluation,
  getModelRegistry,
  rollbackModel,
  getSensitivity,
  type Evaluation,
  type ModelRegistrySnapshot,
} from "./api";
import Analytics from "./Analytics";

vi.mock("../digit-studio/lib/api", () => ({ predictDigit: vi.fn() }));
vi.mock("../../lib/current-model-input", () => ({ readCurrentModelInput: vi.fn() }));
vi.mock("./api", () => ({
  activateModel: vi.fn(),
  compareModels: vi.fn(),
  evaluateCheckpoint: vi.fn(),
  getComparisonHistory: vi.fn(),
  getComparisonReports: vi.fn(),
  getErrorImage: vi.fn(),
  getEvaluation: vi.fn(),
  getModelRegistry: vi.fn(),
  rollbackModel: vi.fn(),
  getSensitivity: vi.fn(),
}));

const pixels = Array.from({ length: 28 }, () => Array<number>(28).fill(0));
const report: Evaluation = {
  model_version: "test-model", checkpoint_sha256: "a".repeat(64), dataset: "MNIST", dataset_split: "official-test",
  dataset_images_sha256: "b".repeat(64), dataset_labels_sha256: "c".repeat(64),
  generated_at: "2026-01-01", accuracy: 0.9, loss: 0.1, num_samples: 10,
  confusion_matrix: Array.from({ length: 10 }, (_, row) => Array.from({ length: 10 }, (_, col) => row === col ? 1 : 0)),
  precision_per_class: Array<number>(10).fill(1), recall_per_class: Array<number>(10).fill(1),
  f1_per_class: Array<number>(10).fill(1), support_per_class: Array<number>(10).fill(1),
  misclassified_examples: [
    { sample_index: 4, true_label: 2, predicted_label: 7, confidence: 0.8 },
    { sample_index: 5, true_label: 3, predicted_label: 8, confidence: 0.7 },
  ],
};
function registry(activeHash: string | null, checkpointHashes: string[] = [report.checkpoint_sha256]): ModelRegistrySnapshot {
  const checkpoints = checkpointHashes.map((checkpoint_hash, index) => ({
    checkpoint_hash,
    model_version: index === 0 ? "test-model" : "test-model-v2",
    architecture: index === 0 ? "mlp" : "cnn",
    checkpoint_name: index === 0 ? "best.pt" : "models/cnn.pt",
    is_loaded: checkpoint_hash === activeHash,
  }));
  const activeCheckpoint = checkpoints.find((item) => item.checkpoint_hash === activeHash);
  const active = activeCheckpoint ? { ...activeCheckpoint, activated_at: "2026-01-01T00:00:00Z" } : null;
  return {
    active,
    available_checkpoints: checkpoints,
    history: active ? [active] : [],
  };
}

afterEach(() => { cleanup(); vi.resetAllMocks(); });

describe("Analytics", () => {
  it("filters actual errors, loads a selected image, and changes sensitivity target", async () => {
    vi.mocked(getEvaluation).mockResolvedValue(report);
    vi.mocked(evaluateCheckpoint).mockResolvedValue(report);
    vi.mocked(getModelRegistry).mockResolvedValue(registry(report.checkpoint_sha256));
    vi.mocked(getComparisonReports).mockResolvedValue([report]);
    vi.mocked(getComparisonHistory).mockResolvedValue([]);
    vi.mocked(readCurrentModelInput).mockReturnValue(null);
    vi.mocked(getErrorImage).mockResolvedValue(pixels);
    vi.mocked(predictDigit).mockResolvedValue({
      model_id: "mnist-mlp-v1", model_version: "test-model", predicted_class: 7,
      probabilities: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], logits: Array<number>(10).fill(0), inference_ms: 1.2,
    });
    vi.mocked(getSensitivity).mockImplementation(async (_pixels, target) => ({
      model_version: "test-model", predicted_class: 7, target_class: target ?? 7,
      probabilities: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], gradient: pixels,
    }));
    render(<Analytics />);
    expect(await screen.findByText("90.00%")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Counts by true and predicted digit" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("True digit"), { target: { value: "2" } });
    expect(screen.getByText("1 matching errors")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /#5/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /#4/ }));
    expect(await screen.findByText("MNIST test sample 4")).toBeInTheDocument();
    expect(getErrorImage).toHaveBeenCalledWith(4, report.checkpoint_sha256, expect.any(AbortSignal));
    const region = screen.getByRole("region", { name: "Prediction and sensitivity" });
    expect(await within(region).findByText(/inference 1.20 ms/)).toBeInTheDocument();
    expect(await within(region).findByText(/Class 7 logit/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Sensitivity target"), { target: { value: "2" } });
    expect(await within(region).findByText(/Class 2 logit/)).toBeInTheDocument();
    expect(within(region).getByText(/not a definitive explanation/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Compare evaluated checkpoints" })).toBeInTheDocument();
  });

  it("shows unavailable data and missing current input without fabricated results", async () => {
    vi.mocked(getEvaluation).mockRejectedValue(new Error("No matching report"));
    vi.mocked(getModelRegistry).mockResolvedValue(registry(null, []));
    vi.mocked(getComparisonReports).mockResolvedValue([]);
    vi.mocked(getComparisonHistory).mockResolvedValue([]);
    vi.mocked(readCurrentModelInput).mockReturnValue(null);
    render(<Analytics />);
    expect(await screen.findByRole("alert")).toHaveTextContent("No matching report");
    fireEvent.click(screen.getByRole("button", { name: "Load current Digit Studio input" }));
    expect(screen.getByText(/No current digit in this tab/)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(predictDigit).not.toHaveBeenCalled();
  });

  it("compares two saved evaluation reports and displays the persisted result", async () => {
    const secondReport = { ...report, model_version: "test-model-v2", checkpoint_sha256: "d".repeat(64) };
    const comparisonResult = {
      comparison_id: "comparison-1",
      model_a_name: "test-model",
      model_b_name: "test-model-v2",
      model_a_version: "test-model",
      model_b_version: "test-model-v2",
      compared_at: "2026-01-02T00:00:00Z",
      summary: "test-model-v2 improved accuracy.",
      metrics: [{
        metric_name: "Overall Accuracy",
        model_a_value: 0.9,
        model_b_value: 0.95,
        difference: 0.05,
        percent_change: 5.55,
      }],
    };
    vi.mocked(getEvaluation).mockResolvedValue(report);
    vi.mocked(evaluateCheckpoint).mockResolvedValue(report);
    vi.mocked(getModelRegistry).mockResolvedValue(registry(report.checkpoint_sha256, [
      report.checkpoint_sha256,
      secondReport.checkpoint_sha256,
    ]));
    vi.mocked(getComparisonReports).mockResolvedValue([report, secondReport]);
    vi.mocked(getComparisonHistory).mockResolvedValue([]);
    vi.mocked(compareModels).mockResolvedValue(comparisonResult);
    vi.mocked(readCurrentModelInput).mockReturnValue(null);
    render(<Analytics />);

    expect(await screen.findByText("2 stored evaluation reports")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Compare checkpoints" }));
    expect(await screen.findByText("test-model-v2 improved accuracy.")).toBeInTheDocument();
    expect(compareModels).toHaveBeenCalledWith(report, secondReport);
    expect(screen.getAllByRole("table").length).toBeGreaterThan(1);
  });

  it("evaluates only a selected checkpoint through the backend", async () => {
    const secondCheckpoint = {
      checkpoint_hash: "d".repeat(64), model_version: "test-model-v2",
      architecture: "cnn", checkpoint_name: "models/cnn.pt", is_loaded: false,
    };
    vi.mocked(getEvaluation).mockResolvedValue(report);
    vi.mocked(evaluateCheckpoint).mockResolvedValue(report);
    vi.mocked(getModelRegistry).mockResolvedValue(registry(report.checkpoint_sha256, [
      report.checkpoint_sha256,
      secondCheckpoint.checkpoint_hash,
    ]));
    vi.mocked(getComparisonReports).mockResolvedValue([report]);
    vi.mocked(getComparisonHistory).mockResolvedValue([]);
    vi.mocked(readCurrentModelInput).mockReturnValue(null);
    render(<Analytics />);

    fireEvent.change(await screen.findByLabelText("Approved checkpoint"), {
      target: { value: secondCheckpoint.checkpoint_hash },
    });
    fireEvent.click(screen.getByRole("button", { name: "Evaluate checkpoint on official test split" }));
    expect(evaluateCheckpoint).toHaveBeenLastCalledWith(secondCheckpoint.checkpoint_hash);
  });

  it("activates and rolls back persisted model versions", async () => {
    const secondReport = { ...report, model_version: "test-model-v2", checkpoint_sha256: "d".repeat(64) };
    const checkpoints = [report.checkpoint_sha256, secondReport.checkpoint_sha256];
    const secondRegistry = registry(secondReport.checkpoint_sha256, checkpoints);
    secondRegistry.history.push({
      ...registry(report.checkpoint_sha256).active!,
    });
    vi.mocked(getEvaluation).mockResolvedValueOnce(report).mockResolvedValueOnce(secondReport).mockResolvedValue(report);
    vi.mocked(getModelRegistry).mockResolvedValue(registry(report.checkpoint_sha256, checkpoints));
    vi.mocked(getComparisonReports).mockResolvedValue([report, secondReport]);
    vi.mocked(getComparisonHistory).mockResolvedValue([]);
    vi.mocked(activateModel).mockResolvedValue(secondRegistry);
    vi.mocked(rollbackModel).mockResolvedValue(registry(report.checkpoint_sha256, checkpoints));
    vi.mocked(readCurrentModelInput).mockReturnValue(null);
    render(<Analytics />);

    fireEvent.change(await screen.findByLabelText("Checkpoint to activate"), {
      target: { value: secondReport.checkpoint_sha256 },
    });
    fireEvent.click(screen.getByRole("button", { name: "Activate checkpoint" }));
    expect(await screen.findByText("Activated model: test-model-v2.")).toBeInTheDocument();
    expect(activateModel).toHaveBeenCalledWith(secondReport.checkpoint_sha256);

    fireEvent.click(screen.getByRole("button", { name: "Roll back to previous model" }));
    expect(await screen.findByText("Rolled back to: test-model.")).toBeInTheDocument();
    expect(rollbackModel).toHaveBeenCalled();
  });
});
