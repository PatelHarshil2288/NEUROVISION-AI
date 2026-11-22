"""Performance optimization utilities and monitoring."""

from __future__ import annotations

import time
import logging
from functools import wraps
from typing import Any, Callable, TypeVar

import torch

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class PerformanceMetrics:
    """Track performance metrics for API operations."""

    def __init__(self):
        self.operation_times: dict[str, list[float]] = {}
        self.memory_snapshots: dict[str, list[int]] = {}

    def record_operation(self, operation_name: str, elapsed_time: float) -> None:
        """Record operation execution time."""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        self.operation_times[operation_name].append(elapsed_time)

    def get_average_time(self, operation_name: str) -> float | None:
        """Get average execution time for an operation."""
        times = self.operation_times.get(operation_name, [])
        return sum(times) / len(times) if times else None

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all operation metrics."""
        summary = {}
        for op_name, times in self.operation_times.items():
            if times:
                summary[op_name] = {
                    "average_ms": (sum(times) / len(times)) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                    "count": len(times),
                }
        return summary


def profile_operation(operation_name: str) -> Callable[[F], F]:
    """Decorator to profile operation execution time."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start
                logger.debug(
                    f"{operation_name} completed in {elapsed*1000:.2f}ms"
                )
        return wrapper  # type: ignore
    return decorator


def optimize_inference(model: torch.nn.Module, input_shape: tuple[int, ...]) -> None:
    """Apply inference optimizations to a model.
    
    Args:
        model: PyTorch model to optimize
        input_shape: Expected input tensor shape (without batch dimension)
    """
    # Set to evaluation mode
    model.eval()
    
    # Disable gradients for inference
    torch.set_grad_enabled(False)
    
    # Optionally enable cuDNN benchmarking for CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
        # Use tf32 for performance on NVIDIA GPUs
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True


def batch_inference(
    model: torch.nn.Module,
    inputs: list[torch.Tensor],
    batch_size: int = 32,
) -> list[torch.Tensor]:
    """Run inference on a list of inputs in batches for efficiency.
    
    Args:
        model: PyTorch model
        inputs: List of input tensors
        batch_size: Batch size for inference
        
    Returns:
        List of output tensors in original order
    """
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = torch.stack(inputs[i : i + batch_size])
            batch_outputs = model(batch)
            outputs.extend(batch_outputs.split(1))
    
    return [output.squeeze(0) for output in outputs]


def get_model_memory_usage(model: torch.nn.Module) -> int:
    """Get estimated memory usage of model parameters in bytes."""
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_params


class InferenceCache:
    """LRU cache for inference results."""

    def __init__(self, max_size: int = 128):
        self.cache: dict[str, Any] = {}
        self.access_order: list[str] = []
        self.max_size = max_size

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[key] = value
        self.access_order.append(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()

    def size(self) -> int:
        """Get number of items in cache."""
        return len(self.cache)
