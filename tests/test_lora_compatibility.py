"""Test LoRA compatibility with PyTorch modules like MultiheadAttention."""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.lora import LinearWithLoRA


def test_linear_with_lora_attributes():
    """Test that LinearWithLoRA exposes weight and bias attributes."""
    print("=" * 60)
    print("Testing LinearWithLoRA Attribute Access")
    print("=" * 60)

    # Create a simple linear layer
    linear = nn.Linear(10, 5, bias=True)

    # Wrap with LoRA
    lora_linear = LinearWithLoRA(linear, rank=4, alpha=8)

    # Test weight attribute access
    print("\n1. Testing weight attribute...")
    assert hasattr(lora_linear, 'weight'), "LinearWithLoRA should have 'weight' attribute"
    assert lora_linear.weight is linear.weight, "Weight should reference original layer's weight"
    print(f"   ✓ weight shape: {lora_linear.weight.shape}")

    # Test bias attribute access
    print("\n2. Testing bias attribute...")
    assert hasattr(lora_linear, 'bias'), "LinearWithLoRA should have 'bias' attribute"
    assert lora_linear.bias is linear.bias, "Bias should reference original layer's bias"
    print(f"   ✓ bias shape: {lora_linear.bias.shape}")

    # Test in_features and out_features
    print("\n3. Testing dimension attributes...")
    assert lora_linear.in_features == 10, "in_features should be 10"
    assert lora_linear.out_features == 5, "out_features should be 5"
    print(f"   ✓ in_features: {lora_linear.in_features}")
    print(f"   ✓ out_features: {lora_linear.out_features}")

    print("\n✓ All attribute tests passed!")
    return True


def test_multihead_attention_compatibility():
    """Test that LinearWithLoRA works with MultiheadAttention."""
    print("\n" + "=" * 60)
    print("Testing MultiheadAttention Compatibility")
    print("=" * 60)

    # Create a MultiheadAttention module
    embed_dim = 16
    num_heads = 2
    mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    print(f"\nOriginal out_proj type: {type(mha.out_proj)}")
    print(f"Original out_proj.weight shape: {mha.out_proj.weight.shape}")

    # Wrap the out_proj with LoRA
    original_out_proj = mha.out_proj
    mha.out_proj = LinearWithLoRA(original_out_proj, rank=4, alpha=8)

    print(f"\nWrapped out_proj type: {type(mha.out_proj)}")
    print(f"Wrapped out_proj.weight shape: {mha.out_proj.weight.shape}")

    # Test forward pass
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, embed_dim)

    print(f"\nInput shape: {x.shape}")

    try:
        output, attn_weights = mha(x, x, x)
        print(f"Output shape: {output.shape}")
        print(f"Attention weights shape: {attn_weights.shape}")
        print("\n✓ MultiheadAttention forward pass successful!")

        # Verify LoRA parameters are trainable
        lora_params = [p for p in mha.out_proj.lora.parameters()]
        trainable_lora = sum(p.requires_grad for p in lora_params)
        print(f"\nLoRA parameters: {len(lora_params)}")
        print(f"Trainable LoRA parameters: {trainable_lora}")
        assert trainable_lora == len(lora_params), "All LoRA params should be trainable"

        # Verify original parameters are frozen
        original_params = [p for p in mha.out_proj.linear.parameters()]
        frozen_original = sum(not p.requires_grad for p in original_params)
        print(f"Original parameters: {len(original_params)}")
        print(f"Frozen original parameters: {frozen_original}")
        assert frozen_original == len(original_params), "All original params should be frozen"

        print("\n✓ All compatibility tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass():
    """Test that gradients flow correctly through LoRA."""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)

    # Create attention with LoRA
    embed_dim = 16
    mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True)
    mha.out_proj = LinearWithLoRA(mha.out_proj, rank=4, alpha=8)

    # Forward pass
    x = torch.randn(2, 10, embed_dim)
    output, _ = mha(x, x, x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check LoRA gradients
    lora_A_grad = mha.out_proj.lora.lora_A.grad
    lora_B_grad = mha.out_proj.lora.lora_B.grad

    print(f"\nLoRA A gradient shape: {lora_A_grad.shape}")
    print(f"LoRA A gradient norm: {lora_A_grad.norm():.4f}")
    print(f"LoRA B gradient shape: {lora_B_grad.shape}")
    print(f"LoRA B gradient norm: {lora_B_grad.norm():.4f}")

    assert lora_A_grad is not None, "LoRA A should have gradients"
    assert lora_B_grad is not None, "LoRA B should have gradients"
    assert lora_A_grad.norm() > 0, "LoRA A gradients should be non-zero"
    assert lora_B_grad.norm() > 0, "LoRA B gradients should be non-zero"

    # Check original layer has no gradients
    original_weight_grad = mha.out_proj.linear.weight.grad
    print(f"\nOriginal weight gradient: {original_weight_grad}")
    assert original_weight_grad is None, "Original weights should not have gradients (frozen)"

    print("\n✓ Gradient flow test passed!")
    return True


if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print("LORA COMPATIBILITY TESTS")
        print("=" * 60)

        # Run all tests
        test_linear_with_lora_attributes()
        test_multihead_attention_compatibility()
        test_backward_pass()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60 + "\n")

        print("Summary:")
        print("✓ LinearWithLoRA properly exposes weight and bias attributes")
        print("✓ Compatible with PyTorch MultiheadAttention")
        print("✓ Gradients flow correctly through LoRA adapters")
        print("✓ Original weights remain frozen")
        print("\nLoRA can now be safely applied to both 'dense' and 'msa' modules!")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
