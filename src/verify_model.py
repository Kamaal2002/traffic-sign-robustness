"""Verify the baseline CNN with `python -m src.verify_model`.

Run from the project root with the project virtual environment. This script
checks the approved architecture, parameter counts, intermediate shapes, that
the model emits logits, that eval() disables Dropout, that a backward pass
reaches every trainable tensor, that BatchNorm is configured to track
statistics and does accumulate them in train mode while staying frozen in eval
mode, and that no parameter changes while no optimizer exists. It does not
create an optimizer, take an optimizer step, or train the model, and it does
not touch the official test split.
"""

import argparse

import torch
from torch import nn

from .data import IMAGE_SIZE, NUM_CLASSES, SEED, create_data_loaders
from .model import BLOCK_CHANNELS, CLASSIFIER_WIDTH, BaselineCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    model = BaselineCNN()
    print(f"Instantiated {type(model).__name__} for {NUM_CLASSES} classes "
          f"at {IMAGE_SIZE}x{IMAGE_SIZE} input.")

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # The whole baseline is trained from scratch, so nothing should be frozen.
    # A stray requires_grad_(False) would otherwise train a different model than
    # the one that was approved, and the gradient checks below would simply run
    # over the smaller set without complaining.
    assert trainable_parameters == total_parameters
    print(f"Parameters: {total_parameters:,} total; {trainable_parameters:,} trainable")
    feature_parameters = sum(p.numel() for p in model.features.parameters())
    classifier_parameters = sum(p.numel() for p in model.classifier.parameters())
    print(f"  convolutional blocks: {feature_parameters:,}; "
          f"classifier: {classifier_parameters:,}")
    assert total_parameters == feature_parameters + classifier_parameters

    # The approved classifier is spelled out here on purpose. This script exists
    # to catch accidental changes to a frozen architecture, so encoding the
    # intended sequence is the point rather than a duplication to be avoided.
    # Unpacking below is safe because the layout assertion runs first.
    expected_layout = [nn.Flatten, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear]
    classifier_layout = [type(module) for module in model.classifier]
    assert classifier_layout == expected_layout, classifier_layout
    _, first_linear, _, dropout, final_linear = model.classifier
    spatial = model.image_size // 2 ** len(BLOCK_CHANNELS)
    assert first_linear.in_features == BLOCK_CHANNELS[-1] * spatial * spatial
    assert first_linear.out_features == CLASSIFIER_WIDTH
    assert final_linear.in_features == CLASSIFIER_WIDTH
    assert final_linear.out_features == model.num_classes == NUM_CLASSES
    assert dropout.p == 0.5
    print(f"Classifier: Flatten -> Linear({first_linear.in_features} -> "
          f"{first_linear.out_features}) -> ReLU -> Dropout(p={dropout.p}) -> "
          f"Linear({final_linear.in_features} -> {final_linear.out_features})")

    # BatchNorm configuration, checked before any forward pass. The corruption
    # evaluation runs under eval() on statistics learned during training, so a
    # layer that never tracks them, or one whose momentum has drifted, would
    # undermine that experiment while still producing a trainable model.
    batch_norm_layers = [(name, module) for name, module in model.named_modules()
                         if isinstance(module, nn.BatchNorm2d)]
    assert len(batch_norm_layers) == len(BLOCK_CHANNELS), len(batch_norm_layers)
    for name, layer in batch_norm_layers:
        assert layer.track_running_stats, name
        assert layer.momentum == 0.1, (name, layer.momentum)
        assert layer.running_mean is not None and layer.running_var is not None, name
        assert layer.num_batches_tracked is not None, name
    print(f"All {len(batch_norm_layers)} BatchNorm layers track running statistics "
          f"with momentum 0.1 and own running_mean/running_var buffers")

    # Intermediate shapes, module by module, so the 64 -> 4 reduction is visible
    # rather than inferred.
    model.eval()
    expected_shapes = [(2, 32, 32, 32), (2, 64, 16, 16), (2, 128, 8, 8), (2, 256, 4, 4)]
    activations = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    print(f"Input: {tuple(activations.shape)}")
    with torch.no_grad():
        for index, (block, expected) in enumerate(zip(model.features, expected_shapes), start=1):
            activations = block(activations)
            assert tuple(activations.shape) == expected, (index, activations.shape)
            print(f"  block {index}: {tuple(activations.shape)}")
        for module in model.classifier:
            activations = module(activations)
            print(f"  {type(module).__name__.lower()}: {tuple(activations.shape)}")
        assert tuple(activations.shape) == (2, NUM_CLASSES), activations.shape

    # Synthetic batches, including a single image, to confirm the batch
    # dimension is never assumed.
    with torch.no_grad():
        for batch in (1, 8):
            outputs = model(torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE))
            assert outputs.shape == (batch, NUM_CLASSES)
            assert torch.isfinite(outputs).all()
            print(f"Synthetic batch {batch}: {tuple(outputs.shape)} logits, all finite")

    # The classifier must emit logits. A Softmax anywhere in the model would
    # double-normalise under CrossEntropyLoss, so check structurally and
    # numerically rather than by reading the source.
    forbidden = (nn.Softmax, nn.LogSoftmax, nn.Softmax2d, nn.Sigmoid)
    assert not [m for m in model.modules() if isinstance(m, forbidden)]
    with torch.no_grad():
        probe = model(torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE))
    row_sums = probe.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert (probe < 0).any() and probe.max() > 0
    print(f"No Softmax/Sigmoid module in the model; logit row sums span "
          f"[{row_sums.min():.3f}, {row_sums.max():.3f}] and values are signed")

    # eval() must disable Dropout, because the corruption evaluation compares
    # models under deployment-style inference and a stochastic forward would
    # make those numbers irreproducible. Only the Dropout module's mode is
    # switched here: the rest of the model, BatchNorm included, stays in eval,
    # so a difference between the two groups below can only come from Dropout.
    dropout_probe = torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        deterministic = [model(dropout_probe) for _ in range(2)]
        dropout.train()
        stochastic = [model(dropout_probe) for _ in range(3)]
        dropout.eval()
        restored = model(dropout_probe)
    assert torch.equal(*deterministic), "eval-mode forwards are not deterministic"
    assert not any(torch.equal(stochastic[0], other) for other in stochastic[1:]), \
        "Dropout is inactive in train mode"
    assert not torch.equal(stochastic[0], deterministic[0])
    assert torch.equal(restored, deterministic[0])
    print(f"Dropout(p={dropout.p}) is inert under eval(): two eval forwards on one "
          f"batch are identical, three train-mode forwards all differ")

    # A configurable class count must actually change the output width.
    assert BaselineCNN(num_classes=10)(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)).shape == (2, 10)
    print("num_classes is configurable; num_classes=10 produced (2, 10)")

    # A real batch from the existing pipeline, to confirm the model and the data
    # loader agree on shape, dtype, and class range. Train and validation only;
    # the official test split stays untouched.
    loaders, _ = create_data_loaders(batch_size=args.batch_size, download=False)
    for split in ("train", "validation"):
        images, labels = next(iter(loaders[split]))
        assert images.shape == (args.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            outputs = model(images)
        assert outputs.shape == (args.batch_size, NUM_CLASSES)
        assert outputs.dtype == torch.float32
        assert torch.isfinite(outputs).all()
        assert labels.min() >= 0 and labels.max() < NUM_CLASSES
        print(f"Real {split} batch {tuple(images.shape)} -> {tuple(outputs.shape)} logits, "
              f"all finite, range [{outputs.min():.3f}, {outputs.max():.3f}]")

    # Backward pass only. Every trainable tensor should receive a gradient that
    # exists, is finite, and is not entirely zero. These are three distinct
    # failures, not one: a detached path leaves gradients absent, a zeroed path
    # leaves them present and finite but identically zero, and a numerically
    # unstable path leaves them non-finite. Collapsing them hides whichever one
    # occurred, so each is detected and reported separately.
    model.train()
    parameters_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    buffers_before = {name: b.detach().clone() for name, b in model.named_buffers()}
    assert len(buffers_before) == 3 * len(batch_norm_layers), len(buffers_before)
    images, labels = next(iter(loaders["train"]))
    backward_images, backward_labels = images[:8], labels[:8]
    loss = nn.CrossEntropyLoss()(model(backward_images), backward_labels)
    assert torch.isfinite(loss)
    loss.backward()

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == len(parameters_before)
    no_gradient = [name for name, p in trainable if p.grad is None]
    non_finite = [name for name, p in trainable if p.grad is not None
                  and not torch.isfinite(p.grad).all()]
    entirely_zero = [name for name, p in trainable if p.grad is not None
                     and torch.isfinite(p.grad).all() and not p.grad.any()]
    assert not no_gradient, f"no gradient reached: {no_gradient}"
    assert not non_finite, f"non-finite gradient: {non_finite}"
    assert not entirely_zero, f"gradient entirely zero: {entirely_zero}"
    gradient_norm = torch.sqrt(sum((p.grad ** 2).sum() for _, p in trainable))
    print(f"Backward pass on {len(backward_images)} samples: loss {loss.item():.4f}, "
          f"gradient norm {gradient_norm:.4f}")
    print(f"  all {len(trainable)} trainable tensors received a gradient that is "
          f"present, finite, and not entirely zero")

    # BatchNorm running statistics must genuinely accumulate during training,
    # because the corruption evaluation later runs under model.eval() on the
    # statistics learned here. num_batches_tracked increments on every
    # train-mode forward whether or not the statistics move, so it is excluded
    # from the evidence and counted separately.
    changed_buffers = [name for name, b in model.named_buffers()
                       if not torch.equal(buffers_before[name], b)]
    changed_statistics = [name for name in changed_buffers
                          if name.endswith(("running_mean", "running_var"))]
    changed_counters = [name for name in changed_buffers
                        if name.endswith("num_batches_tracked")]
    # Per layer, requiring both statistics to move would be too strong: a
    # channel whose activations are identically zero leaves running_mean at
    # zero while running_var still moves. Requiring at least one per layer is
    # what the update rule actually guarantees.
    without_update = [name for name, _ in batch_norm_layers
                      if not any(stat.startswith(f"{name}.") for stat in changed_statistics)]
    assert not without_update, f"running statistics did not update: {without_update}"
    assert len(changed_counters) == len(batch_norm_layers)
    print(f"  the train-mode forward updated the running statistics of all "
          f"{len(batch_norm_layers)} BatchNorm layers "
          f"({len(changed_statistics)} running_mean/running_var tensors changed; "
          f"{len(changed_counters)} num_batches_tracked counters incremented, "
          f"which is not counted as evidence)")

    # Under eval() the same forward must leave every buffer alone. This is the
    # deployment-style inference the corruption evaluation depends on, so the
    # frozen behaviour is asserted rather than assumed.
    model.eval()
    frozen_before = {name: b.detach().clone() for name, b in model.named_buffers()}
    assert len(frozen_before) == 3 * len(batch_norm_layers), len(frozen_before)
    with torch.no_grad():
        model(backward_images)
    moved_in_eval = [name for name, b in model.named_buffers()
                     if not torch.equal(frozen_before[name], b)]
    assert not moved_in_eval, f"eval-mode forward changed buffers: {moved_in_eval}"
    print(f"  an eval-mode forward left all {len(frozen_before)} buffers unchanged")

    # Parameters are what an optimizer would change, and none was created, so
    # they must be identical. BatchNorm running statistics are buffers rather
    # than parameters, and were updated by the train-mode forward above.
    changed_parameters = [name for name, p in model.named_parameters()
                          if not torch.equal(parameters_before[name], p)]
    assert not changed_parameters, changed_parameters
    print(f"No optimizer was created and no step was taken; all "
          f"{len(parameters_before)} parameter tensors are unchanged; "
          f"nothing was saved to disk.")
    print("All baseline CNN checks passed.")


if __name__ == "__main__":
    main()
