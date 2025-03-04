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
    p1 = x

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
    p2 = x

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
    p3 = x

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
    p4 = x

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
    p5 = x

    return p1, p2, p3, p4, p5
