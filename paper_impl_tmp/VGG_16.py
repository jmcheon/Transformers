import tensorflow as tf


def block(x, n_convs, filters, kernel_size, activation, pool_size, pool_stride, block_name):
    for i in range(n_convs):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding="same",
            name=f"{block_name}_conv_{i + 1}",
        )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=pool_size, strides=pool_stride, name=f"{block_name}_pool_{i + 1}"
    )(x)

    return x


def VGG_16_conv(image_input):
    # Conv_1
    x = block(
        image_input,
        n_convs=2,
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        pool_size=(2, 2),
        pool_stride=(2, 2),
        block_name="block1",
    )

    # Conv_2
    x = block(
        x,
        n_convs=2,
        filters=128,
        kernel_size=(3, 3),
        activation="relu",
        pool_size=(2, 2),
        pool_stride=(2, 2),
        block_name="block2",
    )

    # Conv_3
    x = block(
        x,
        n_convs=3,
        filters=256,
        kernel_size=(3, 3),
        activation="relu",
        pool_size=(2, 2),
        pool_stride=(2, 2),
        block_name="block3",
    )

    # Conv_4
    x = block(
        x,
        n_convs=3,
        filters=512,
        kernel_size=(3, 3),
        activation="relu",
        pool_size=(2, 2),
        pool_stride=(2, 2),
        block_name="block4",
    )

    # Conv_5
    x = block(
        x,
        n_convs=3,
        filters=512,
        kernel_size=(3, 3),
        activation="relu",
        pool_size=(2, 2),
        pool_stride=(2, 2),
        block_name="block5",
    )

    return x


def VGG_16_fcn(x, n_classes):
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation="relu", name="fc2")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax", name="prediction")(x)

    return x


def VGG_16(x, n_classes):
    x = VGG_16_conv(x)
    x = VGG_16_fcn(x, n_classes)

    return x
