import jax.numpy as jnp
import defaults_sghmc as defaults
from jax.experimental.host_callback import call as jcall
import os

debug = os.environ.get("DEBUG")

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
features_mean = jnp.array(
    [0.18330987, 0.7001979,  0.6445455,  0.7460847,  0.5094299,  0.6295954,
     0.6442109,  0.8646987, 0.9002692, 1.0193347, 0.51987416, 0.68272555,
     0.81302977, 0.19370939, 0.41179654, 0.6538645, 0.9568495, 0.6743199,
     0.60995865, 0.8898805, 0.679904, 0.84155005, 0.910891, 0.09549496,
     1.2345159, 0.68164, 0.8379528, 0.59505486, 0.10757514, 0.65255636,
     0.16895305, 1.0504391, 0.06856065, 0.26042646, 0.92757004, 0.3574842,
     0.47494194, 0.8827974, 0.18126468, 0.16119045, 0.62743354, 0.4908466,
     0.28185168, 0.7584322, 0.22031961, 0.78558546, 0.8418701, 0.75707096,
     1.2195462, 0.38055083, 0.8270162, 0.37286556, 0.63944095, 0.7566279,
     0.6003291, 0.19767928, 0.800389, 0.7437937, 0.76633406, 0.4014488,
     0.7003062, 0.4731737, 0.4598344, 0.37230557])
features_std = jnp.array(
    [0.14133751, 0.45575073, 0.41797432, 0.5406982, 0.41299394, 0.49650577,
     0.5040387, 0.5003279, 0.99623924, 0.7080817, 0.41074216, 0.5239645,
     0.53119606, 0.22514634, 0.46265996, 0.6806917, 0.6417554, 0.5902017,
     0.5533807, 0.5918419, 0.5663299, 0.68205875, 1.1546439, 0.06593046,
     0.9975034, 0.6358142, 0.6969466, 0.77471215, 0.10301507, 0.6556335,
     0.15704316, 0.5918194, 0.19334291, 0.3057011, 0.73842627, 0.23153019,
     0.34169877, 0.60623777, 0.13861355, 0.24570253, 0.8521112, 0.40651947,
     0.35084727, 0.6528583, 0.23616803, 0.9620372, 0.5736422, 0.6876864,
     0.6879774, 0.25772202, 0.5487339, 0.38053706, 0.56402767, 0.63259363,
     0.4913096, 0.1632212, 0.717361, 0.5536584, 0.6357068, 0.2597183,
     0.5801118, 0.46481824, 0.47353822, 0.31616378])


def normalize(x):
    return (x-pixel_mean)/pixel_std
    # return (x-pixel_mean)


def unnormalize(x):
    return pixel_std*x+pixel_mean
    # return x+pixel_mean


def normalize_logits(x, features_dir="features_last"):
    if features_dir == "features":
        mean = logits_mean
        std = logits_std
    elif features_dir == "features_last":
        mean = features_mean
        std = features_std
    return (x-mean)/std


def unnormalize_logits(x, features_dir="features_last"):
    if features_dir == "features":
        mean = logits_mean
        std = logits_std
    elif features_dir == "features_last":
        mean = features_mean
        std = features_std
    return mean+std*x


def pixelize(x):
    return (x*255).astype("uint8")


def jprint_fn(*args):
    fstring = ""
    arrays = []
    for i, a in enumerate(args):
        if i != 0:
            fstring += " "
        if isinstance(a, str):
            fstring += a
        else:
            fstring += '{}'
            arrays.append(a)

    jcall(lambda arrays: print(fstring.format(*arrays)), arrays)


jprint = lambda *args: ...
if debug:
    jprint = jprint_fn
