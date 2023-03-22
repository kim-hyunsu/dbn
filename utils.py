import jax.numpy as jnp
import defaults_sghmc as defaults

model_list = [
    "./checkpoints/frn_sd2",
    "./checkpoints/frn_sd3",
    "./checkpoints/frn_sd5",
    "./checkpoints/frn_sd7",
    "./checkpoints/frn_sd11",
    "./checkpoints/frn_sd13",
    "./checkpoints/frn_sd17"
]


pixel_mean = jnp.array(defaults.PIXEL_MEAN, dtype=jnp.float32)
pixel_std = jnp.array(defaults.PIXEL_STD, dtype=jnp.float32)

logits_mean = jnp.array(
    [0.43181, 0.7272722, -0.18803911, 0.196443, -1.4037232, 0.38403097, -0.5623876, -0.30297565, -0.6744309, 1.3405691])
logits_std = jnp.array(
    [5.718929, 6.735006, 5.898479, 5.6934276, 5.23119,  5.5628223, 5.522649, 6.565273, 6.072292, 6.3432117])


def normalize(x):
    # return (x-pixel_mean)/pixel_std
    return (x-pixel_mean)


def unnormalize(x):
    # return pixel_std*x+pixel_mean
    return x+pixel_mean


def normalize_logits(x):
    return (x-logits_mean)/logits_std


def unnormalize_logits(x):
    return logits_mean+logits_std*x


def pixelize(x):
    return (x*255).astype("uint8")
