import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import TrainingLab from "./TrainingLab";
import { getTrainingStatus, startTraining, type TrainingStatus } from "./api";

vi.mock("./api", () => ({
  getTrainingStatus: vi.fn(),
  startTraining: vi.fn(),
}));

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe("TrainingLab", () => {
  it("explains disabled training without permitting a job", async () => {
    vi.mocked(getTrainingStatus).mockResolvedValue({ enabled: false, remaining_jobs: 3, job: null });
    render(<TrainingLab />);
    expect(await screen.findByText("Local training is disabled.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled();
  });

  it("submits the selected epoch count and displays real epoch and gradient values", async () => {
    const finished: TrainingStatus = {
      enabled: true, remaining_jobs: 2,
      job: {
        id: "job", status: "completed", epochs: 2, epoch: 2, batch: 16, batches: 16,
        train_loss: 0.5, validation_loss: 0.7, validation_accuracy: 0.8,
        optimizer: "Adam", learning_rate: 0.001, train_samples: 2048, validation_samples: 512,
        gradient_norm: 1.2, sample_gradient: -0.3, sample_before: 0.2, sample_after: 0.201,
        history: [{ epoch: 1, train_loss: 1.2, validation_loss: 1.1, validation_accuracy: 0.6 },
          { epoch: 2, train_loss: 0.5, validation_loss: 0.7, validation_accuracy: 0.8 }],
        checkpoint: { latest_epoch: 2, best_epoch: 2, directory: "artifacts/training-lab" },
        error: null,
      },
    };
    vi.mocked(getTrainingStatus)
      .mockResolvedValueOnce({ enabled: true, remaining_jobs: 3, job: null })
      .mockResolvedValue(finished);
    vi.mocked(startTraining).mockResolvedValue();
    render(<TrainingLab />);
    const button = await screen.findByRole("button", { name: "Start training" });
    fireEvent.change(screen.getByLabelText("Epochs"), { target: { value: "2" } });
    fireEvent.click(button);
    await waitFor(() => expect(startTraining).toHaveBeenCalledWith(2));
    expect(await screen.findByText(/COMPLETED/)).toBeInTheDocument();
    expect(screen.getAllByText("1.20000").length).toBeGreaterThan(0);
    expect(screen.getByText(/Latest checkpoint: epoch 2/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Training and validation loss/ })).toBeInTheDocument();
  });
});
