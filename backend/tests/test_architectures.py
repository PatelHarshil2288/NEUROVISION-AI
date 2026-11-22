"""Tests for alternative neural network architectures."""

import torch
import pytest

from neurovision_api.ml.architectures import MNISTConv, MNISTRnn, MNISTTransformer


class TestMNISTConv:
    """Tests for CNN architecture."""

    def test_forward_pass(self):
        """Test CNN forward pass."""
        model = MNISTConv()
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()

    def test_get_activations(self):
        """Test activation extraction."""
        model = MNISTConv()
        x = torch.randn(2, 1, 28, 28)
        activations = model.get_activations(x)
        
        expected_keys = {"conv1", "conv1_relu", "conv2", "conv2_relu", "fc1", "fc1_relu", "output"}
        assert set(activations.keys()) == expected_keys
        
        # Check shapes
        assert activations["conv1"].shape == (2, 32, 28, 28)
        assert activations["output"].shape == (2, 10)

    def test_dropout_behavior(self):
        """Test dropout is disabled in eval mode."""
        model = MNISTConv(dropout_p=0.5)
        model.eval()
        
        x = torch.randn(4, 1, 28, 28)
        
        # Same output in eval mode
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        assert torch.allclose(output1, output2)


class TestMNISTRnn:
    """Tests for RNN architecture."""

    def test_forward_pass(self):
        """Test RNN forward pass."""
        model = MNISTRnn()
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()

    def test_unidirectional_rnn(self):
        """Test unidirectional RNN."""
        model = MNISTRnn(bidirectional=False)
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (2, 10)

    def test_different_hidden_sizes(self):
        """Test RNN with different hidden sizes."""
        for hidden_size in [32, 64, 128]:
            model = MNISTRnn(hidden_size=hidden_size)
            x = torch.randn(2, 1, 28, 28)
            output = model(x)
            assert output.shape == (2, 10)

    def test_different_num_layers(self):
        """Test RNN with different numbers of layers."""
        for num_layers in [1, 2, 3]:
            model = MNISTRnn(num_layers=num_layers)
            x = torch.randn(2, 1, 28, 28)
            output = model(x)
            assert output.shape == (2, 10)


class TestMNISTTransformer:
    """Tests for Transformer architecture."""

    def test_forward_pass(self):
        """Test Transformer forward pass."""
        model = MNISTTransformer()
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()

    def test_positional_encoding_shape(self):
        """Test positional encoding."""
        model = MNISTTransformer(d_model=128)
        assert model.positional_encoding.shape == (784, 128)

    def test_different_d_model(self):
        """Test Transformer with different d_model."""
        for d_model in [64, 128, 256]:
            model = MNISTTransformer(d_model=d_model)
            x = torch.randn(2, 1, 28, 28)
            output = model(x)
            assert output.shape == (2, 10)

    def test_gradient_flow(self):
        """Test that gradients flow through Transformer."""
        model = MNISTTransformer()
        x = torch.randn(2, 1, 28, 28, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


@pytest.mark.parametrize("model_class", [MNISTConv, MNISTRnn, MNISTTransformer])
def test_all_models_inference_speed(model_class):
    """Test inference speed of all models."""
    model = model_class()
    model.eval()
    
    x = torch.randn(1, 1, 28, 28)
    
    with torch.no_grad():
        import time
        start = time.time()
        for _ in range(100):
            _ = model(x)
        elapsed = time.time() - start
    
    # Should complete 100 inferences reasonably fast
    # (depends on hardware, but should be < 1 second on modern CPU)
    assert elapsed < 10.0


@pytest.mark.parametrize("model_class", [MNISTConv, MNISTRnn, MNISTTransformer])
def test_all_models_parameter_count(model_class):
    """Test that models have reasonable parameter counts."""
    model = model_class()
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # All models should have between 10K and 500K parameters
    assert 10_000 < total_params < 500_000
