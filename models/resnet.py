# Revised from https://github.com/cs-giung/giung2-dev/tree/main/giung2/models/resnet.py
import inspect
import functools
from typing import Any, Tuple, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn


class FlaxResNetClassifier(nn.Module):
    """
    For transfering only the last layer of the ResNet
    """
    num_classes: int = None
    dtype: Any = jnp.float32
    fc: nn.Module = functools.partial(nn.Dense, use_bias=True,
                                      kernel_init=jax.nn.initializers.he_normal(),
                                      bias_init=jax.nn.initializers.zeros)

    @nn.compact
    def __call__(self, x, **kwargs):
        return self.fc(features=self.num_classes, dtype=self.dtype)(x)


class FlaxResNetClassifier2(nn.Module):
    """
    For transfering only the second last layer of the ResNet
    """
    num_classes: int = None
    dtype: Any = jnp.float32
    fc: nn.Module = functools.partial(nn.Dense, use_bias=True,
                                      kernel_init=jax.nn.initializers.he_normal(),
                                      bias_init=jax.nn.initializers.zeros)

    @nn.compact
    def __call__(self, x, **kwargs):
        x = jnp.mean(x, axis=(1, 2))
        return self.fc(features=self.num_classes, dtype=self.dtype)(x)


class FlaxResNetClassifier3(nn.Module):
    depth:        int = 20
    widen_factor: float = 1.
    dtype:        Any = jnp.float32
    pixel_mean:   Tuple[int] = (0.0, 0.0, 0.0)
    pixel_std:    Tuple[int] = (1.0, 1.0, 1.0)
    num_classes:  int = None
    conv:         nn.Module = functools.partial(nn.Conv, use_bias=False,
                                                kernel_init=jax.nn.initializers.he_normal(),
                                                bias_init=jax.nn.initializers.zeros)
    norm:         nn.Module = functools.partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                                scale_init=jax.nn.initializers.ones,
                                                bias_init=jax.nn.initializers.zeros)
    relu:         Callable = nn.relu
    fc:           nn.Module = functools.partial(nn.Dense, use_bias=True,
                                                kernel_init=jax.nn.initializers.he_normal(),
                                                bias_init=jax.nn.initializers.zeros)

    @nn.compact
    def __call__(self, x, **kwargs):

        # NOTE: it should be False during training, if we use batch normalization...
        use_running_average = kwargs.pop('use_running_average', True)
        if 'use_running_average' in inspect.signature(self.norm).parameters:
            self.norm.keywords['use_running_average'] = use_running_average

        # NOTE: it should be False during training, if we use batch normalization...
        deterministic = kwargs.pop('deterministic', True)
        if 'deterministic' in inspect.signature(self.conv).parameters:
            self.conv.keywords['deterministic'] = deterministic
        if 'deterministic' in inspect.signature(self.fc).parameters:
            self.fc.keywords['deterministic'] = deterministic

        # batchnorm_kwargs = dict(
        #     use_running_average=not kwargs["training"]
        # ) if isinstance(self.norm, nn.BatchNorm) else dict()

        # standardize input images...
        # m = self.variable('image_stats', 'm', lambda _: jnp.array(
        #     self.pixel_mean, dtype=jnp.float32), (x.shape[-1],))
        # s = self.variable('image_stats', 's', lambda _: jnp.array(
        #     self.pixel_std, dtype=jnp.float32), (x.shape[-1],))
        # x = x - jnp.reshape(m.value, (1, 1, 1, -1))
        # x = x / jnp.reshape(s.value, (1, 1, 1, -1))

        # specify block structure and widen factor...
        num_planes = 16
        num_blocks = [(self.depth - 2) // 6,] * 3
        widen_factor = self.widen_factor

        # define the first layer...
        # y = self.conv(
        #     features=num_planes,
        #     kernel_size=(3, 3),
        #     strides=(1, 1),
        #     padding='SAME',
        #     dtype=self.dtype,
        # )(x)
        # y = self.norm(dtype=self.dtype)(y)
        # y = self.relu(y)
        # self.sow('intermediates', 'feature.layer0', y)
        y = x

        # define intermediate layers...
        for layer_idx, num_block in enumerate(num_blocks):
            _strides = (1,) if layer_idx == 0 else (2,)
            _strides = _strides + (1,) * (num_block - 1)
            if layer_idx != len(num_blocks) - 1:
                continue
            for _stride_idx, _stride in enumerate(_strides, start=1):
                _channel = num_planes * (2 ** layer_idx)
                residual = y

                if _stride_idx < len(_strides)-2:
                    continue
                y = self.conv(
                    features=int(_channel * widen_factor),
                    kernel_size=(3, 3),
                    strides=(_stride, _stride),
                    padding='SAME',
                    dtype=self.dtype,
                )(y)
                y = self.norm(
                    dtype=self.dtype,
                    # **batchnorm_kwargs
                )(y)
                y = self.relu(y)
                y = self.conv(
                    features=int(_channel * widen_factor),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dtype=self.dtype,
                )(y)
                y = self.norm(
                    dtype=self.dtype,
                    # **batchnorm_kwargs
                )(y)
                if residual.shape != y.shape:
                    # NOTE : we use the projection shortcut regardless of the input size,
                    #        which can make a difference compared to He et al. (2016).
                    residual = self.conv(
                        features=int(_channel * widen_factor),
                        kernel_size=(1, 1),
                        strides=(_stride, _stride),
                        padding='SAME',
                        dtype=self.dtype,
                    )(residual)
                    residual = self.norm(
                        dtype=self.dtype,
                        # **batchnorm_kwargs
                    )(residual)

                y = self.relu(y + residual)

        y = jnp.mean(y, axis=(1, 2))

        # return logits if possible
        if self.num_classes:
            y = self.fc(features=self.num_classes, dtype=self.dtype)(y)

        return y


class FlaxResNet(nn.Module):
    depth:        int = 20
    widen_factor: float = 1.
    dtype:        Any = jnp.float32
    pixel_mean:   Tuple[int] = (0.0, 0.0, 0.0)
    pixel_std:    Tuple[int] = (1.0, 1.0, 1.0)
    num_classes:  int = None
    conv:         nn.Module = functools.partial(nn.Conv, use_bias=False,
                                                kernel_init=jax.nn.initializers.he_normal(),
                                                bias_init=jax.nn.initializers.zeros)
    norm:         nn.Module = functools.partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                                scale_init=jax.nn.initializers.ones,
                                                bias_init=jax.nn.initializers.zeros)
    relu:         Callable = nn.relu
    fc:           nn.Module = functools.partial(nn.Dense, use_bias=True,
                                                kernel_init=jax.nn.initializers.he_normal(),
                                                bias_init=jax.nn.initializers.zeros)
    num_planes: int = 16
    num_blocks: Tuple[int] = None

    @nn.compact
    def __call__(self, x, **kwargs):

        # NOTE: it should be False during training, if we use batch normalization...
        use_running_average = kwargs.pop('use_running_average', True)
        if 'use_running_average' in inspect.signature(self.norm).parameters:
            self.norm.keywords['use_running_average'] = use_running_average

        # NOTE: it should be False during training, if we use batch normalization...
        deterministic = kwargs.pop('deterministic', True)
        if 'deterministic' in inspect.signature(self.conv).parameters:
            self.conv.keywords['deterministic'] = deterministic
        if 'deterministic' in inspect.signature(self.fc).parameters:
            self.fc.keywords['deterministic'] = deterministic

        # standardize input images...
        m = self.variable('image_stats', 'm', lambda _: jnp.array(
            self.pixel_mean, dtype=jnp.float32), (x.shape[-1],))
        s = self.variable('image_stats', 's', lambda _: jnp.array(
            self.pixel_std, dtype=jnp.float32), (x.shape[-1],))
        x = x - jnp.reshape(m.value, (1, 1, 1, -1))
        x = x / jnp.reshape(s.value, (1, 1, 1, -1))

        # specify block structure and widen factor...
        num_planes = self.num_planes
        num_blocks = (
            [(self.depth - 2) // 6,] * 3
        ) if self.num_blocks is None else self.num_blocks
        widen_factor = self.widen_factor

        # define the first layer...
        y = self.conv(
            features=num_planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            dtype=self.dtype,
        )(x)
        y = self.norm(dtype=self.dtype)(y)
        y = self.relu(y)
        self.sow('intermediates', 'feature.layer0', y)

        # define intermediate layers...
        for layer_idx, num_block in enumerate(num_blocks):
            _strides = (1,) if layer_idx == 0 else (2,)
            _strides = _strides + (1,) * (num_block - 1)
            for _stride_idx, _stride in enumerate(_strides, start=1):
                _channel = num_planes * (2 ** layer_idx)
                residual = y

                y = self.conv(
                    features=int(_channel * widen_factor),
                    kernel_size=(3, 3),
                    strides=(_stride, _stride),
                    padding='SAME',
                    dtype=self.dtype,
                )(y)
                y = self.norm(dtype=self.dtype)(y)
                y = self.relu(y)
                y = self.conv(
                    features=int(_channel * widen_factor),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dtype=self.dtype,
                )(y)
                y = self.norm(dtype=self.dtype)(y)
                if residual.shape != y.shape:
                    # NOTE : we use the projection shortcut regardless of the input size,
                    #        which can make a difference compared to He et al. (2016).
                    residual = self.conv(
                        features=int(_channel * widen_factor),
                        kernel_size=(1, 1),
                        strides=(_stride, _stride),
                        padding='SAME',
                        dtype=self.dtype,
                    )(residual)
                    residual = self.norm(dtype=self.dtype)(residual)

                if _stride_idx == len(_strides):
                    self.sow('intermediates',
                             f'pre_relu_feature.layer{layer_idx + 1}', y)
                else:
                    self.sow('intermediates',
                             f'pre_relu_feature.layer{layer_idx + 1}stride{_stride_idx}', y)

                y = self.relu(y + residual)

                if _stride_idx == len(_strides):
                    self.sow('intermediates',
                             f'feature.layer{layer_idx + 1}', y)
                else:
                    self.sow('intermediates',
                             f'feature.layer{layer_idx + 1}stride{_stride_idx}', y)

        y = jnp.mean(y, axis=(1, 2))
        self.sow('intermediates', 'feature.vector', y)

        # return logits if possible
        if self.num_classes:
            y = self.fc(features=self.num_classes, dtype=self.dtype)(y)
            self.sow('intermediates', 'cls.logit', y)

        return y


class FlaxResNetBase(nn.Module):
    depth:        int = 20
    widen_factor: float = 1.
    dtype:        Any = jnp.float32
    pixel_mean:   Tuple[int] = (0.0, 0.0, 0.0)
    pixel_std:    Tuple[int] = (1.0, 1.0, 1.0)
    num_classes:  int = None
    conv:         nn.Module = functools.partial(nn.Conv, use_bias=False,
                                                kernel_init=jax.nn.initializers.he_normal(),
                                                bias_init=jax.nn.initializers.zeros)
    norm:         nn.Module = functools.partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                                scale_init=jax.nn.initializers.ones,
                                                bias_init=jax.nn.initializers.zeros)
    relu:         Callable = nn.relu
    fc:           nn.Module = functools.partial(nn.Dense, use_bias=True,
                                                kernel_init=jax.nn.initializers.he_normal(),
                                                bias_init=jax.nn.initializers.zeros)
    out: str = "feature.layer3stride2"

    @nn.compact
    def __call__(self, x, **kwargs):

        # NOTE: it should be False during training, if we use batch normalization...
        use_running_average = kwargs.pop('use_running_average', True)
        if 'use_running_average' in inspect.signature(self.norm).parameters:
            self.norm.keywords['use_running_average'] = use_running_average

        # NOTE: it should be False during training, if we use batch normalization...
        deterministic = kwargs.pop('deterministic', True)
        if 'deterministic' in inspect.signature(self.conv).parameters:
            self.conv.keywords['deterministic'] = deterministic
        if 'deterministic' in inspect.signature(self.fc).parameters:
            self.fc.keywords['deterministic'] = deterministic

        batchnorm_kwargs = dict(
            use_running_average=not kwargs["training"]
        ) if isinstance(self.norm, nn.BatchNorm) else dict()

        # standardize input images...
        m = self.variable('image_stats', 'm', lambda _: jnp.array(
            self.pixel_mean, dtype=jnp.float32), (x.shape[-1],))
        s = self.variable('image_stats', 's', lambda _: jnp.array(
            self.pixel_std, dtype=jnp.float32), (x.shape[-1],))
        x = x - jnp.reshape(m.value, (1, 1, 1, -1))
        x = x / jnp.reshape(s.value, (1, 1, 1, -1))

        # specify block structure and widen factor...
        num_planes = 16
        num_blocks = [(self.depth - 2) // 6,] * 3
        widen_factor = self.widen_factor

        # define the first layer...
        y = self.conv(
            features=num_planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            dtype=self.dtype,
        )(x)
        y = self.norm(
            dtype=self.dtype,
            use_running_average=not kwargs["training"]
        )(y)
        y = self.relu(y)
        feature_name = "feature.layer0"
        self.sow('intermediates', feature_name, y)
        if feature_name == self.out:
            return y

        # define intermediate layers...
        for layer_idx, num_block in enumerate(num_blocks):
            _strides = (1,) if layer_idx == 0 else (2,)
            _strides = _strides + (1,) * (num_block - 1)
            for _stride_idx, _stride in enumerate(_strides, start=1):
                _channel = num_planes * (2 ** layer_idx)
                residual = y

                y = self.conv(
                    features=int(_channel * widen_factor),
                    kernel_size=(3, 3),
                    strides=(_stride, _stride),
                    padding='SAME',
                    dtype=self.dtype,
                )(y)
                y = self.norm(
                    dtype=self.dtype,
                    use_running_average=not kwargs["training"]
                )(y)
                y = self.relu(y)
                y = self.conv(
                    features=int(_channel * widen_factor),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dtype=self.dtype,
                )(y)
                y = self.norm(
                    dtype=self.dtype,
                    use_running_average=not kwargs["training"]
                )(y)
                if residual.shape != y.shape:
                    # NOTE : we use the projection shortcut regardless of the input size,
                    #        which can make a difference compared to He et al. (2016).
                    residual = self.conv(
                        features=int(_channel * widen_factor),
                        kernel_size=(1, 1),
                        strides=(_stride, _stride),
                        padding='SAME',
                        dtype=self.dtype,
                    )(residual)
                    residual = self.norm(
                        dtype=self.dtype,
                        use_running_average=not kwargs["training"]
                    )(residual)

                if _stride_idx == len(_strides):
                    feature_name = f'pre_relu_feature.layer{layer_idx + 1}'
                    self.sow('intermediates', feature_name, y)
                else:
                    feature_name = f'pre_relu_feature.layer{layer_idx + 1}stride{_stride_idx}'
                    self.sow('intermediates', feature_name, y)
                if feature_name == self.out:
                    return y

                y = self.relu(y + residual)

                if _stride_idx == len(_strides):
                    feature_name = f'feature.layer{layer_idx + 1}'
                    self.sow('intermediates', feature_name, y)
                else:
                    feature_name = f'feature.layer{layer_idx + 1}stride{_stride_idx}'
                    self.sow('intermediates', feature_name, y)
                if feature_name == self.out:
                    return y

        y = jnp.mean(y, axis=(1, 2))
        feature_name = 'feature.vector'
        self.sow('intermediates', feature_name, y)
        if feature_name == self.out:
            return y

        # return logits if possible
        if self.num_classes:
            feature_name = 'cls.logit'
            y = self.fc(features=self.num_classes, dtype=self.dtype)(y)
            self.sow('intermediates', feature_name, y)
            if feature_name == self.out:
                return y

        return y
