from typing import Callable
import dataclasses

import tensorflow as tf


def hardswish(x):
    return x * tf.nn.relu6(x + 3) / 6.0


class SqueezeAndExcitation(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.se = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(input_channels // 4, activation="relu"),
                tf.keras.layers.Dense(input_channels, activation="hard_sigmoid"),
                tf.keras.layers.Reshape([1, 1, input_channels]),
            ]
        )

    def call(self, x, training=False):
        y = self.se(x, training=training)
        return x * y


class MBConvBlock(tf.keras.layers.Layer):
    @dataclasses.dataclass
    class Config:
        out_size: int
        exp_size: int
        use_se: bool
        activation: Callable[[tf.Tensor], tf.Tensor]
        kernel_size: int
        strides: int = 1

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.pointwise_conv = tf.keras.layers.Conv2D(
            config.exp_size, kernel_size=1, strides=(1, 1), padding="same"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=config.kernel_size,
            strides=(config.strides, config.strides),
            padding="same",
        )
        self.use_residual_connection = config.strides == 1
        self.activation = config.activation
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SqueezeAndExcitation() if config.use_se else None
        self.conv = tf.keras.layers.Conv2D(config.out_size, kernel_size=1)
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        y = x
        # pointwise
        y = self.pointwise_conv(y, training=training)
        y = self.bn1(y, training=training)
        y = self.activation(y)
        # depthwise
        y = self.depthwise_conv(y, training=training)
        y = self.bn2(y, training=training)
        y = self.activation(y)
        # se
        if self.se is not None:
            y = self.se(y)
        # pointwise
        y = self.conv(y, training=training)
        y = self.bn3(y, training=training)
        y = self.activation(y)
        # residual
        if self.use_residual_connection and x.shape[-1] == y.shape[-1]:
            y = x + y
        return y


class MobileNetV3(tf.keras.Model):
    def __init__(self, input_shape=None, include_top=True, classes=1000):
        super().__init__()
        self.conv_bn_stem = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    16, kernel_size=3, strides=(2, 2), padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        self.activation_stem = hardswish
        block_configs = [
            MBConvBlock.Config(
                out_size=16,
                exp_size=16,
                use_se=True,
                activation=tf.nn.relu,
                kernel_size=3,
                strides=2,
            ),
            MBConvBlock.Config(
                out_size=24,
                exp_size=72,
                use_se=False,
                activation=tf.nn.relu,
                kernel_size=3,
                strides=2,
            ),
            MBConvBlock.Config(
                out_size=24,
                exp_size=88,
                use_se=False,
                activation=tf.nn.relu,
                kernel_size=3,
                strides=1,
            ),
            MBConvBlock.Config(
                out_size=40,
                exp_size=96,
                use_se=True,
                activation=hardswish,
                kernel_size=5,
                strides=2,
            ),
            MBConvBlock.Config(
                out_size=40,
                exp_size=240,
                use_se=True,
                activation=hardswish,
                kernel_size=5,
                strides=1,
            ),
            MBConvBlock.Config(
                out_size=40,
                exp_size=240,
                use_se=True,
                activation=hardswish,
                kernel_size=5,
                strides=1,
            ),
            MBConvBlock.Config(
                out_size=48,
                exp_size=120,
                use_se=True,
                activation=hardswish,
                kernel_size=5,
                strides=1,
            ),
            MBConvBlock.Config(
                out_size=48,
                exp_size=144,
                use_se=True,
                activation=hardswish,
                kernel_size=5,
                strides=1,
            ),
            MBConvBlock.Config(
                out_size=96,
                exp_size=288,
                use_se=True,
                activation=hardswish,
                kernel_size=5,
                strides=2,
            ),
            MBConvBlock.Config(
                out_size=96,
                exp_size=576,
                use_se=True,
                activation=hardswish,
                kernel_size=5,
                strides=1,
            ),
            MBConvBlock.Config(
                out_size=96,
                exp_size=576,
                use_se=True,
                activation=hardswish,
                kernel_size=5,
                strides=1,
            ),
        ]
        self.blocks = tf.keras.Sequential(
            [MBConvBlock(config) for config in block_configs]
        )
        self.conv_bn_feat = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(576, kernel_size=1),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        self.activation_feat = hardswish
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(1280),
            tf.keras.layers.Lambda(hardswish),
            tf.keras.layers.Dense(classes),
        ])

    def call(self, x, training=False):
        x = self.conv_bn_stem(x, training=training)
        x = self.activation_stem(x)
        x = self.blocks(x, training=training)
        x = self.conv_bn_feat(x, training=training)
        x = self.activation_feat(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.classifier(x, training=training)
        return x
