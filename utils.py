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


def normalize(x):
    return (x-pixel_mean)/pixel_std


def unnormalize(x):
    return pixel_std*x+pixel_mean


def pixelize(x):
    return (x*255).astype("uint8")
