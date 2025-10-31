"""Test script for learning rate scheduler."""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.scheduler import WarmupCosineScheduler, create_scheduler


def test_warmup_cosine_scheduler():
    """Test WarmupCosineScheduler implementation."""
    print("=" * 60)
    print("Testing WarmupCosineScheduler")
    print("=" * 60)

    # Create a simple model and optimizer
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=5,
        max_epochs=50,
        warmup_start_lr=1e-7,
        min_lr=1e-6
    )

    print("\nLearning rate schedule (first 20 epochs):")
    print("Epoch | Learning Rate | Phase")
    print("-" * 45)

    for epoch in range(20):
        lr = scheduler.get_last_lr()[0]
        phase = "Warmup" if epoch < 5 else "Cosine Annealing"
        print(f"{epoch:5d} | {lr:.2e}         | {phase}")
        scheduler.step()

    print("\nWarmupCosineScheduler test PASSED!")
    return True


def test_scheduler_factory():
    """Test create_scheduler factory function."""
    print("\n" + "=" * 60)
    print("Testing create_scheduler Factory")
    print("=" * 60)

    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Test warmup_cosine
    print("\n1. Testing 'warmup_cosine' type...")
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type='warmup_cosine',
        max_epochs=100,
        warmup_epochs=10,
        min_lr=1e-6,
        warmup_start_lr=1e-8
    )
    print(f"   Created: {type(scheduler).__name__}")
    assert isinstance(scheduler, WarmupCosineScheduler), "Should create WarmupCosineScheduler"

    # Test cosine
    print("\n2. Testing 'cosine' type...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type='cosine',
        max_epochs=100,
        min_lr=1e-6
    )
    print(f"   Created: {type(scheduler).__name__}")
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    # Test step
    print("\n3. Testing 'step' type...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type='step',
        max_epochs=100,
        step_size=30,
        gamma=0.1
    )
    print(f"   Created: {type(scheduler).__name__}")
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    # Test none
    print("\n4. Testing 'none' type...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type='none',
        max_epochs=100
    )
    print(f"   Created: {scheduler}")
    assert scheduler is None, "Should return None for type 'none'"

    print("\nScheduler factory test PASSED!")
    return True


def test_scheduler_values():
    """Test that scheduler produces expected learning rate values."""
    print("\n" + "=" * 60)
    print("Testing Scheduler Values")
    print("=" * 60)

    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=5,
        max_epochs=10,
        warmup_start_lr=0.0,
        min_lr=0.0
    )

    # Test warmup phase (should increase linearly)
    print("\nWarmup phase (epochs 0-4):")
    for epoch in range(5):
        lr = scheduler.get_last_lr()[0]
        expected_lr = 0.001 * epoch / 5
        print(f"  Epoch {epoch}: LR = {lr:.6f}, Expected ≈ {expected_lr:.6f}")
        assert abs(lr - expected_lr) < 1e-6, f"Warmup LR mismatch at epoch {epoch}"
        scheduler.step()

    # Test cosine phase
    print("\nCosine annealing phase (epochs 5-9):")
    for epoch in range(5, 10):
        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch}: LR = {lr:.6f}")
        scheduler.step()

    print("\nScheduler values test PASSED!")
    return True


if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print("LEARNING RATE SCHEDULER TESTS")
        print("=" * 60)

        # Run all tests
        test_warmup_cosine_scheduler()
        test_scheduler_factory()
        test_scheduler_values()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
