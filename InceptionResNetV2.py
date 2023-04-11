from tensorflow.keras import layers 

def conv2d_bn(
    x,
    filters,
    kernel_size,
    strides=1,
    padding="same",
    activation="relu",
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
    )(x)
    x = layers.BatchNormalization(axis=3, scale=False)(x)
    x = layers.Activation("relu")(x)
    return x

def inception_resnet_block(x, scale, block_type, block_idx, activation=):

    if block_type == "block35":
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == "block17":
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == "block8":
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError("Unknown Inception-ResNet block type. ")

    mixed = layers.Concatenate(axis=3, name= block_type + "_" + str(block_idx) + "_mixed")(branches)
    up = conv2d_bn(
        mixed,
        backend.int_shape(x)[channel_axis],
        1,
        activation=None,
        use_bias=True,
        name=block_type + "_" + str(block_idx) + "_conv",
    )

    x = CustomScaleLayer(scale)([x, up])
    x = layers.Activation("relu", name=block_type + "_" + str(block_idx) + "_ac")(x)
    return x

def InceptionResNetV2(
    include_top=True,
    input_shape=None,
    pooling=None,
    classes=100,
    classifier_activation="softmax",
):

    img_input = layers.Input(shape=input_shape)

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding="valid")
    x = conv2d_bn(x, 32, 3, padding="valid")
    x = conv2d_bn(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding="valid")
    x = conv2d_bn(x, 192, 3, padding="valid")
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]

    x = layers.Concatenate(axis=3, name="mixed_5b")(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x, scale=0.17, block_type="block35", block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding="valid")
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding="valid")
    branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name="mixed_6a")(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x, scale=0.1, block_type="block17", block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding="valid")
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding="valid")
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding="valid")
    branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name="mixed_7a")(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x, scale=0.2, block_type="block8", block_idx=block_idx)
    x = inception_resnet_block(x, scale=1.0, activation=None, block_type="block8", block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name="conv_7b")

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation=classifier_activation, name="predictions")(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    model = training.Model(img_input, x, name="inception_resnet_v2")

    return model












