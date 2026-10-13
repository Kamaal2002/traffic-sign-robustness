"""Custom CNN baseline for GTSRB traffic-sign classification."""

from torch import nn

from .data import IMAGE_SIZE, NUM_CLASSES

# Channel width of each convolutional block. Each block halves the spatial
# resolution, so 64x64 inputs reach 4x4 after the four blocks.
BLOCK_CHANNELS = (32, 64, 128, 256)
CLASSIFIER_WIDTH = 256


def conv_block(in_channels, out_channels):
    """Conv -> BatchNorm -> ReLU -> MaxPool, halving the spatial resolution.

    A 3x3 kernel with padding 1 leaves the spatial size unchanged, so pooling is
    the only thing that changes resolution and the block's output shape stays
    easy to follow. The convolution has no bias term because the BatchNorm that
    follows applies its own shift, which would make a conv bias redundant.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class BaselineCNN(nn.Module):
    """Four-block CNN over 64x64 RGB traffic signs, returning raw class logits.

    Channels widen as resolution shrinks (3x64x64 -> 32x32x32 -> 64x16x16 ->
    128x8x8 -> 256x4x4), which is the usual trade of spatial detail for feature
    capacity. Four blocks is enough depth to build up from edges and colour to
    whole-glyph structure while keeping the model small and easy to explain.

    `forward` returns unnormalised logits. No Softmax is applied, because
    `nn.CrossEntropyLoss` expects logits and applies log-softmax internally;
    adding Softmax here would double-normalise and distort the gradients.

    BatchNorm and Dropout are both deliberate. BatchNorm stabilises training and
    keeps this baseline strong enough that later augmentation gains cannot be
    dismissed as repairs to a weak model. Dropout guards the classifier, where
    most of the parameters sit, against overfitting 21,300 training images.
    """

    def __init__(self, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, dropout=0.5):
        super().__init__()

        # One pooling layer per block, so the input must stay divisible by 2 all
        # the way down rather than silently truncating to a smaller feature map.
        # Non-positive sizes are rejected separately, because 0 and -16 satisfy
        # the divisibility test and would otherwise be misreported.
        downsampling = 2 ** len(BLOCK_CHANNELS)
        if image_size <= 0:
            raise ValueError(f"image_size must be positive; got {image_size}")
        if image_size % downsampling != 0:
            raise ValueError(
                f"image_size must be divisible by {downsampling}; got {image_size}"
            )
        spatial = image_size // downsampling

        # Recorded so a live module can report the shape it was built for.
        # Plain ints are not part of state_dict(), so a weights-only checkpoint
        # cannot recover them; they will be saved explicitly alongside the model
        # state when training is implemented.
        self.num_classes = num_classes
        self.image_size = image_size

        self.features = nn.Sequential(
            conv_block(3, BLOCK_CHANNELS[0]),
            conv_block(BLOCK_CHANNELS[0], BLOCK_CHANNELS[1]),
            conv_block(BLOCK_CHANNELS[1], BLOCK_CHANNELS[2]),
            conv_block(BLOCK_CHANNELS[2], BLOCK_CHANNELS[3]),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BLOCK_CHANNELS[-1] * spatial * spatial, CLASSIFIER_WIDTH),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(CLASSIFIER_WIDTH, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
