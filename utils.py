import jax.numpy as jnp
import defaults_sghmc as defaults
from jax.experimental.host_callback import call as jcall
import os

debug = os.environ.get("DEBUG")
if isinstance(debug, str):
    debug = debug.lower() == "true"

model_list = [
    "./checkpoints/frn_sd2",
    "./checkpoints/frn_sd3",
    "./checkpoints/frn_sd5",
    "./checkpoints/frn_sd7",
    "./checkpoints/frn_sd11",
    "./checkpoints/frn_sd13",
    "./checkpoints/frn_sd17"
]
logit_dir_list = ["features", "features_fixed",
                  "features_1mixup", "features100"]
feature_dir_list = ["features_last", "features_last_fixed",
                    "features_last_1mixup", "features100_last"]


pixel_mean = jnp.array(defaults.PIXEL_MEAN, dtype=jnp.float32)
pixel_std = jnp.array(defaults.PIXEL_STD, dtype=jnp.float32)

logits_mean = jnp.array(
    [0.20203184, 0.6711326, -0.34316424, 0.55240935, -1.6872929, 0.47505122, -0.44700608, -0.33773288, -0.5277096, 1.1016572])
logits_std = jnp.array(
    [6.0323553, 6.419274, 5.948161, 5.245664,  5.7556744, 5.959587, 5.682561, 6.168531, 6.233299, 5.9158483])
logits_fixed_mean = jnp.array(
    [-0.31134173, -0.25409052, 0.03164461, 0.6108086, -0.7115754, 0.45521107, -0.05110987, -0.17799538, -0.7254453, 0.8062728])
logits_fixed_std = jnp.array(
    [6.245104, 6.566702, 6.3997235, 5.580838, 6.454161, 6.315527, 5.996594, 6.4479523, 6.5356264, 6.2433324])
features_mean = jnp.array(
    [0.18330994, 0.7001978, 0.6445457, 0.7460846, 0.50943, 0.62959504,
     0.64421076, 0.864699, 0.90026885, 1.0193346, 0.5198741, 0.6827253,
     0.8130299, 0.19370948, 0.4117968, 0.65386456, 0.9568493, 0.6743198,
     0.60995907, 0.8898808, 0.6799041, 0.84155, 0.910891, 0.09549502,
     1.2345164, 0.68164027, 0.8379526, 0.59505486, 0.10757514, 0.65255636,
     0.16895305, 1.0504391, 0.06856067, 0.2604265, 0.92756987, 0.35748425,
     0.4749423, 0.88279736, 0.18126479, 0.1611905, 0.6274334, 0.4908467,
     0.28185168, 0.7584321, 0.22031951, 0.7855851, 0.8418701, 0.7570713,
     1.2195468, 0.38055098, 0.8270158, 0.3728654, 0.6394407, 0.75662774,
     0.6003293, 0.1976792, 0.8003896, 0.74379367, 0.7663342, 0.4014488,
     0.7003062, 0.4731737, 0.4598342, 0.37230548])
features_std = jnp.array(
    [0.14625898, 0.46195114, 0.43259677, 0.54840773, 0.41533348, 0.47113135,
     0.48780078, 0.52398956, 1.0137664, 0.6777552, 0.42684877, 0.5078425,
     0.5339083, 0.22176242, 0.48372495, 0.67294866, 0.6393872, 0.61134773,
     0.5523887, 0.58954775, 0.5916741, 0.7449703, 1.1613424, 0.0671833,
     0.96865875, 0.6145824, 0.7023753, 0.7894523, 0.10271957, 0.650923,
     0.15482505, 0.60614026, 0.19371748, 0.30993605, 0.72206664, 0.23388785,
     0.3628471, 0.6144786, 0.13806942, 0.25634378, 0.76641285, 0.44256744,
     0.3434019, 0.6321352, 0.23825505, 0.915923, 0.6375949, 0.7456085,
     0.71694154, 0.25504288, 0.5668632, 0.41129625, 0.5674771, 0.6080663,
     0.5312087, 0.1603583, 0.71235883, 0.56369525, 0.6138001, 0.26981887,
     0.6191205, 0.5023357, 0.5107374, 0.33516783])
features_fixed_mean = jnp.array(
    [0.18705866, 0.68849486, 0.6488158, 0.78207123, 0.5511557, 0.6262307,
     0.65450567, 0.8826836, 0.8655664, 1.00992, 0.54281944, 0.7618934,
     0.7957989, 0.21154717, 0.43999764, 0.68076986, 0.89898854, 0.684323,
     0.6646593, 0.95220345, 0.69163674, 0.9322681, 0.8517829, 0.09536795,
     1.2630196, 0.70337015, 1.012143, 0.6002648, 0.11455201, 0.64685863,
     0.17385696, 1.023957, 0.05679391, 0.27782413, 0.9697278, 0.37249184,
     0.4667875, 0.92745036, 0.16634183, 0.18392234, 0.6530218, 0.52148014,
     0.3018203, 0.7654548, 0.22229634, 0.7955025, 0.8523904, 0.77817434,
     1.2886604, 0.37336883, 0.81984186, 0.38767225, 0.66403496, 0.7473745,
     0.6249284, 0.20447044, 0.8548401, 0.7030146, 0.7588822, 0.41101518,
     0.748774, 0.48774925, 0.5226807, 0.38154706])
features_fixed_std = jnp.array(
    [0.15364195, 0.47394925, 0.45756358, 0.57014096, 0.47625965, 0.4805353,
     0.50975424, 0.56288433, 1.0426552, 0.68454623, 0.46168298, 0.5579592,
     0.5530582, 0.24337238, 0.5236689, 0.71482545, 0.65052545, 0.6139082,
     0.5803394, 0.61604536, 0.60494155, 0.7878116, 1.1275738, 0.06792182,
     0.9790216, 0.6222661, 0.7810331, 0.7999032, 0.10657259, 0.65028626,
     0.15862404, 0.62626255, 0.1890049, 0.33651996, 0.7579544, 0.23396464,
     0.370818, 0.6821918, 0.14029975, 0.27957192, 0.7865221, 0.4593404,
     0.35550362, 0.6629405, 0.24056178, 0.956299, 0.6544769, 0.7641251,
     0.7616899, 0.26205933, 0.58135045, 0.42836696, 0.59881216, 0.6225946,
     0.5618599, 0.16933021, 0.7180478, 0.57931596, 0.62415534, 0.27320555,
     0.6263999, 0.5018518, 0.56142384, 0.35803506])
logits_1mixup_mean = jnp.array([])
logits_1mixup_std = jnp.array([])
features_1mixup_mean = jnp.array([])
features_1mixup_std = jnp.array([])
logits100_mean = jnp.array([])
logits100_std = jnp.array([])
features100_mean = jnp.array([])
features100_std = jnp.array([])


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
    elif features_dir == "features_fixed":
        mean = logits_fixed_mean
        std = logits_fixed_std
    elif features_dir == "features_last":
        mean = features_mean
        std = features_std
    elif features_dir == "features_last_fixed":
        mean = features_fixed_mean
        std = features_fixed_std
    elif features_dir == "features100":
        mean = logits100_mean
        std = logits100_std
    elif features_dir == "features100_last":
        mean = features100_mean
        std = features100_std
    return (x-mean)/std


def unnormalize_logits(x, features_dir="features_last"):
    if features_dir == "features":
        mean = logits_mean
        std = logits_std
    elif features_dir == "features_fixed":
        mean = logits_fixed_mean
        std = logits_fixed_std
    elif features_dir == "features_last":
        mean = features_mean
        std = features_std
    elif features_dir == "features_last_fixed":
        mean = features_fixed_mean
        std = features_fixed_std
    elif features_dir == "features100":
        mean = logits100_mean
        std = logits100_std
    elif features_dir == "features100_last":
        mean = features100_mean
        std = features100_std
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
    print("**** DEBUG MODE ****")
    jprint = jprint_fn
