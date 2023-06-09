import jax.numpy as jnp
import defaults_sghmc as defaults
from jax.experimental.host_callback import call as jcall
import os
import wandb
import jax
from flax import jax_utils
from flax.training import common_utils

debug = os.environ.get("DEBUG")
if isinstance(debug, str):
    debug = debug.lower() == "true"

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
logits_smooth_mean = jnp.array([
    1.0167354,-0.6048022,   1.3812016  , 1.1861428 , 0.41711354,  0.6843606,0.12473508 , 0.20884748, -0.6354314 ,  0.30543354])
logits_smooth_std = jnp.array([
    2.9246716,4.6175737,2.9916914,3.044312 ,3.7991865,3.5999544,3.4893486,3.8933403,3.799254 ,4.065405 ])
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
features_smooth_mean = jnp.array([2.2591686,1.6846231,2.4262881,1.8620274,2.5296724,1.7403471,2.1779778,
 2.397413 ,1.6612936,2.1530817,1.8401841,1.8240654,2.054809 ,2.099684,
 2.103704 ,2.150928 ,1.8618845,2.1831872,1.8943466,2.2140658,2.1225722,
 2.1992414,1.7928971,2.6273873,1.1930034,1.5750698,2.3884864,2.394339,
 2.2787273,2.2732646,1.8455626,1.5083206,1.8373212,1.8319095,1.2800753,
 2.0343556,1.7147509,2.0220613,2.5702703,2.412537 ,2.0320156,1.733261,
 2.2042649,2.0588052,2.1886444,2.0548208,2.2332573,1.8787292,1.7449883,
 1.3538883,2.196829 ,1.5567267,1.66049  ,1.6265544,2.199724 ,2.4128587,
 1.6366208,1.4242903,1.7274755,1.8444502,1.5184199,2.1950843,1.4311628,
 2.1932156])
features_smooth_std = jnp.array([0.94868135,0.6022823 ,0.94064116,0.91724193,1.0014375 ,0.6991272,
 0.734294  ,1.9525324 ,0.77374303,0.8054996 ,0.60202974,0.7682523,
 1.0824217 ,1.0532749 ,1.0070893 ,0.71122473,1.0078379 ,0.9384319,
 0.783794  ,1.0543379 ,0.9645058 ,0.8237384 ,0.892413  ,1.4607546,
 0.48913634,0.6910001 ,0.7946616 ,0.8736424 ,1.0135208 ,1.1901033,
 0.85582936,0.7762151 ,0.98204595,0.98938817,0.773156  ,0.9948165,
 0.8303378 ,1.2158434 ,1.7871115 ,1.1158894 ,0.84670997,0.9412111,
 1.052403  ,1.00306   ,0.864706  ,0.737708  ,0.9060585 ,0.8076786,
 0.95391244,0.68137455,1.6059061 ,0.6699924 ,0.6911579 ,0.5560083,
 1.1337692 ,0.97876066,0.87733203,0.9236515 ,0.6251136 ,0.95945084,
 0.7923356 ,1.0319482 ,0.5455782 ,0.92253876])
logits_1mixup10_mean = jnp.array([])
logits_1mixup10_std = jnp.array([])
features_1mixup10_mean = jnp.array([])
features_1mixup10_std = jnp.array([])
logits100_mean = jnp.array([-0.78918993,  0.18090598, 0.86517256, 0.30723083, 0.15103784, 0.09180187,
                            -0.44571838, 0.06200991, 0.14822228, 0.20730133, -0.03349496, 0.2144022,
                            -0.22495835, -0.6576749, 0.3294145, 0.42581627, -0.03646952, -0.2745286,
                            0.5139872, 0.13040869, -0.39609373, 0.11353918, -0.09560826, -0.7812039,
                            -0.01596168, 0.6383833, 0.2423121, 0.35803756, -0.694735, 0.30421862,
                            -0.51733416, 0.6392468, 0.9821324, 0.44309995, -0.52617687, 0.7559611,
                            -0.19326943, 0.25391576, 0.26185268, 0.69964933, 0.67458445, 0.4378887,
                            -0.50212127, -0.6836131, 0.81061697, 1.1402507, 0.38741606, -1.2742743,
                            -0.4928542, -0.11932599, 0.73366827, 0.92223614, -1.5527763, -0.8730931,
                            -0.03154067, 0.4191136, 0.16300222, -0.11278426, -0.81230134, -0.3627709,
                            -0.5530198, -0.34061927, -0.43050647, -0.0126726, 0.43686315, 1.0989546,
                            0.1716562, 0.6125122, -0.18166947, -0.16695522, -0.793086, -0.601069,
                            0.31621766, -0.2643169, -0.12333494, 0.45243606, -0.4168915, 0.41478395,
                            0.5238542, -0.15747295, 0.24275906, -0.87225085, 0.21865062, -0.19165191,
                            0.73901576, -0.20824657, -0.21554343, 0.6735802, -0.229494, 0.15955655,
                            0.24073105, -0.00808247, -0.56759256, 0.3187379, -0.6380629, -0.93570644,
                            -0.44494024, -0.46781015, 0.32707706, 0.2370915])
logits100_std = jnp.array([4.594066, 4.5431347, 4.8213224, 4.446335, 4.612145, 4.8464046, 4.6765847,
                           4.540119, 4.3807864, 4.8282657, 4.7882257, 4.864854, 4.9172616, 4.931213,
                           4.542918, 4.1491427, 4.8020263, 4.4095273, 4.38634, 4.5744014, 4.26479,
                           4.218057, 4.463791, 4.4645486, 4.5250025, 4.771777, 4.5610957, 4.3980875,
                           4.8701997, 4.733348, 4.687855, 4.420623, 4.4623585, 4.633992, 4.6156363,
                           4.7556376, 4.590167, 4.3946466, 4.5372133, 4.086129, 4.9869947, 4.7050343,
                           4.8171053, 4.8260055, 4.399883, 4.2015305, 4.504707, 4.7722254, 4.5142837,
                           4.676133, 4.5488057, 4.851657, 4.7159824, 4.786454, 4.5173264, 4.427982,
                           4.704732, 4.9657974, 4.8360343, 4.6786017, 4.585559, 4.3839507, 4.9874907,
                           4.348865, 4.5410476, 4.408959, 4.5149956, 4.479573, 4.287721, 4.653971,
                           4.774027, 4.739601, 4.7271695, 5.0483737, 4.708262, 4.3029838, 4.572212,
                           4.4828134, 4.2437286, 4.6564775, 4.5748825, 4.7701416, 4.6564527, 4.685759,
                           4.545, 4.6726613, 4.753336, 4.603716, 4.613296, 4.608823, 4.528189,
                           4.636887, 4.787973, 4.4409943, 4.425962, 4.6991243, 4.8889084, 4.523881,
                           4.961424, 4.0169554])
features100_mean = jnp.array([2.6821797, 2.1209955, 2.6185288, 2.2039928, 2.5547247, 2.039105, 2.471689,
                              2.1542974, 2.448201, 2.4186301, 1.8662957, 2.9477813, 2.574803, 2.285054,
                              3.3132896, 2.4924183, 2.9913025, 2.3079033, 2.770711, 2.6581316, 3.005827,
                              2.333946, 2.369538, 2.6090517, 2.1910558, 2.622727, 2.634054, 2.8062856,
                              2.5257275, 2.1713817, 2.203236, 2.5777533, 2.497934, 2.4077907, 2.8952234,
                              2.205071, 2.0052783, 2.5821116, 2.7983317, 2.5822134, 2.8375964, 2.5426798,
                              3.2819455, 2.6514902, 2.7889977, 2.2587197, 2.6996057, 2.231013, 2.77432,
                              2.5749366, 2.994471, 2.5471873, 2.4391277, 2.3067093, 2.445581, 2.7716465,
                              2.4797356, 2.9463704, 2.5247407, 2.1275885, 2.3230627, 2.4860115, 2.684688,
                              2.6631765, 2.2830436, 1.9128307, 2.5121222, 2.2116106, 2.4730268, 2.571956,
                              2.4563234, 2.1443346, 2.743497, 2.3312812, 2.6701179, 2.4309192, 2.7292767,
                              3.275847, 2.5726342, 2.1953897, 3.0846527, 2.5292692, 2.675461, 3.1089344,
                              2.3018517, 2.5034602, 2.7797172, 2.4456973, 2.3801384, 2.567207, 2.466297,
                              2.6669574, 2.7068415, 2.7904708, 1.7130059, 2.9451222, 2.728442, 2.3520179,
                              2.2697518, 2.691776, 2.2916102, 2.7537687, 2.288347, 2.289023, 2.352327,
                              2.5175297, 2.746692, 2.0606802, 2.2319562, 2.2945952, 2.4492245, 2.3892665,
                              2.0360563, 2.765004, 2.549863, 2.5857747, 2.539396, 2.451725, 2.6035128,
                              2.9054475, 2.4890354, 2.4755895, 2.5463412, 2.295964, 2.230622, 2.4225528,
                              2.9246604, 2.469591])
features100_std = jnp.array([3.020041, 2.4570334, 2.9119263, 2.7746825, 2.9057117, 2.4719553, 2.6887927,
                             2.5052245, 2.6453543, 2.58863, 2.3927917, 2.9533594, 2.827711, 2.607628,
                             3.222933, 2.9110174, 2.9959304, 2.7891705, 2.8733373, 2.7833514, 3.4780564,
                             2.9139388, 2.6436112, 2.9829261, 2.5115607, 2.9463658, 2.7846432, 3.029591,
                             2.735362, 2.63363, 2.6311672, 2.7553408, 2.7935128, 2.902289, 2.772286,
                             2.6675026, 2.4304438, 2.791632, 3.0617483, 3.291521, 3.0083296, 2.8447104,
                             3.4712183, 3.013435, 2.8616705, 2.7361937, 2.7958577, 2.6078894, 2.9503095,
                             2.8790193, 3.1457403, 2.9656067, 2.6157653, 2.5814729, 2.8496177, 2.9732368,
                             2.7237191, 3.2284594, 2.7750266, 2.4855964, 2.5002184, 2.7398465, 3.1016884,
                             2.8966184, 2.754434, 2.6165552, 2.8638039, 2.6836834, 2.93289, 3.0339184,
                             2.8196006, 2.5376122, 3.1680272, 2.6272118, 2.8788815, 2.7536018, 3.0701041,
                             3.1447372, 3.0442953, 2.5624804, 3.3762753, 3.184333, 3.0205302, 3.189589,
                             2.8517294, 2.680569, 3.1688633, 2.6365554, 2.5802014, 3.0372775, 2.8200786,
                             3.0894337, 3.1149647, 2.9640992, 2.2949948, 3.1538434, 2.9176462, 2.7468798,
                             2.925565, 2.8589933, 2.654712, 2.9383888, 2.4788759, 2.6049254, 2.7213943,
                             2.837745, 2.8747663, 2.5761907, 2.648859, 2.56473, 2.8235097, 2.6415193,
                             2.5395846, 2.8077776, 2.9778378, 2.8533604, 2.7862434, 2.6098804, 3.0832982,
                             3.192695, 2.7096376, 3.0188687, 2.9677248, 2.6474357, 2.6405864, 2.6109276,
                             2.9536164, 2.8505518])


def normalize(x):
    return (x-pixel_mean)/pixel_std
    # return (x-pixel_mean)


def unnormalize(x):
    return pixel_std*x+pixel_mean
    # return x+pixel_mean


def model_list(data_name, model_style, shared_head=False):
    if data_name == "CIFAR100_x32" and model_style == "BN-ReLU":
        if shared_head:
            return [
                "./checkpoints/bn100_sd2_shared",
                "./checkpoints/bn100_sd3_shared",
                "./checkpoints/bn100_sd5_shared",
                "./checkpoints/bn100_sd7_shared",
                "./checkpoints/bn100_sd11_shared",
                "./checkpoints/bn100_sd13_shared",
                "./checkpoints/bn100_sd17_shared",
            ]
        else:
            return [
                "./checkpoints/bn100_sd2",
                "./checkpoints/bn100_sd3",
                "./checkpoints/bn100_sd5",
                "./checkpoints/bn100_sd7",
                "./checkpoints/bn100_sd11",
                "./checkpoints/bn100_sd13",
                "./checkpoints/bn100_sd17",
            ]
    elif data_name == "CIFAR10_x32" and model_style == "FRN-Swish":
        return [
            "./checkpoints/frn_sd2",
            "./checkpoints/frn_sd3",
            "./checkpoints/frn_sd5",
            "./checkpoints/frn_sd7",
            "./checkpoints/frn_sd11",
            "./checkpoints/frn_sd13",
            "./checkpoints/frn_sd17",
            # "./checkpoints/frn_sd19",
        ]
    elif data_name == "CIFAR10_x32" and model_style == "BN-ReLU":
        return [
            # "./checkpoints/bn_sd2_smooth",
            "./checkpoints/bn_sd3_smooth",
            "./checkpoints/bn_sd5_smooth",
            "./checkpoints/bn_sd7_smooth",
        ]
    else:
        raise Exception("Invalid data_name and model_style.")


logit_dir_list = ["features", "features_fixed",
                  "features_1mixup10", "features_1mixup10_fixed",
                  "features_1mixupplus10",
                  "features_smooth",
                  "features100",
                  "features100_ods", "features100_noise",
                  "features100_2ods",
                  "features100_fixed", "features100_shared",
                  "features100_1mixup10", "features100_1mixupplus10",
                  "features100_1mixupext10", "features100_1mixupplusext10",
                  "features100_0p4mixup10_fixed", "features100_0p4mixup10_rand",
                  "features100_0p4mixup10", "features100_0p4mixup10_valid"]
feature_dir_list = ["features_last", "features_last_fixed",
                    "features_last_1mixup10", "features_last_1mixup10_fixed",
                    "features_last_1mixupplus10",
                    "features_last_smooth",
                    "features100_last",
                    "features100_last_1mixup10", "features100_last_0p4mixup10",
                    "features100_last_0p4mixup10_fixed", "features100_last_0p4mixup10_rand",
                    "features100_last_fixed", "features100_last_shared"]
feature2_dir_list = ["features_last2",
                     "features100_last2", "features100_last2_shared"]


def _get_meanstd(features_dir):
    if features_dir == "features":
        mean = logits_mean
        std = logits_std
    elif features_dir in [
            "features_fixed",
            "features_1mixup10",
            "features_1mixup10_fixed",
            "features_1mixupplus10"]:
        mean = logits_fixed_mean
        std = logits_fixed_std
    elif features_dir in ["features_smooth"]:
        mean = logits_smooth_mean
        std = logits_smooth_std
    elif features_dir == "features_last":
        mean = features_mean
        std = features_std
    elif features_dir in ["features_last_smooth"]:
        mean = features_smooth_mean
        std = features_smooth_std
    elif features_dir in ["features_last2"]:
        mean = features_mean[None, None, ...]
        std = features_std[None, None, ...]
    elif features_dir in [
            "features_last_fixed",
            "features_last_1mixup10",
            "features_last_1mixup10_fixed",
            "features_last_1mixupplus10"]:
        mean = features_fixed_mean
        std = features_fixed_std
    elif features_dir in [
        "features100",
        "features100_ods",
        "features100_2ods",
        "features100_noise",
        "features100_fixed",
        "features100_shared",
        "features100_0p4mixup10",
        "features100_1mixup10",
        "features100_0p4mixup10_valid",
        "features100_0p4mixup10_fixed",
        "features100_0p4mixup10_rand",
        "features100_1mixupplus10",
        "features100_1mixupext10",
            "features100_1mixupplusext10"]:
        mean = logits100_mean
        std = logits100_std
    elif features_dir in [
        "features100_last",
        "features100_last_fixed",
        "features100_last_shared",
        "features100_last_0p4mixup10",
        "features100_last_0p4mixup10_fixed",
        "features100_last_0p4mixup10_rand",
            "features100_last_1mixup10"]:
        mean = features100_mean
        std = features100_std
    elif features_dir in ["features100_last2", "features100_last2_shared"]:
        mean = features100_mean[None, None, ...]
        std = features100_std[None, None, ...]
    else:
        raise Exception("Calculate corresponding statistics")
    return mean, std


def normalize_logits(x, features_dir="features_last"):
    mean, std = _get_meanstd(features_dir)
    return (x-mean)/std


def unnormalize_logits(x, features_dir="features_last"):
    mean, std = _get_meanstd(features_dir)
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
else:
    print("**** DEBUG MODE OFF ****")


def get_info_in_dir(dir):
    if "mixupplus" in dir:
        sep = "mixupplus"
    else:
        sep = "mixup"
    if "ext" in dir:
        sep += "ext"
    alpha = float(
        dir.split(sep)[0].split("_")[-1].replace("p", ".")) if sep in dir else -1
    repeats = int(
        dir.split(sep)[1].split("_")[0]) if sep in dir else 1

    return alpha, repeats


class WandbLogger():
    def __init__(self):
        self.summary = dict()
        self.logs = dict()

        self.to_summary = [
            # "trn/acc_ref", "trn/nll_ref", "trn/ens_acc_ref", "trn/ens_t2acc_ref", "trn/ens_nll_ref", "trn/kld_ref", "trn/rkld_ref",
            # "val/acc_ref", "val/nll_ref", "val/ens_acc_ref", "val/ens_t2acc_ref", "val/ens_nll_ref", "val/kld_ref", "val/rkld_ref",
            # "tst/acc_ref", "tst/nll_ref", "tst/ens_acc_ref", "tst/ens_t2acc_ref", "tst/ens_nll_ref", "tst/kld_ref", "tst/rkld_ref",
            "tst/loss",
            "tst/skld", "tst/rkld"
            "tst/sec", "val/sec", "trn/sec"
        ]
        self.summary_keywords = ["ref", "from"]

    def log(self, object):
        for k in self.to_summary:
            value = object.get(k)
            if value is None:
                continue
            self.summary[k] = value
            del object[k]
        for k, v in object.items():
            to_summary = False
            for kw in self.summary_keywords:
                if kw in k:
                    to_summary = True
                    self.summary[k] = v
                    break
            if not to_summary:
                self.logs[k] = v

    def flush(self):
        for k, v in self.summary.items():
            wandb.run.summary[k] = v
        wandb.log(self.logs)
        self.summary = dict()
        self.logs = dict()


def expand_to_broadcast(input, target, axis):
    len_in = len(input.shape)
    len_tar = len(target.shape)
    assert len_tar >= len_in
    expand = len_tar - len_in
    expand = list(range(axis, axis+expand))
    return jnp.expand_dims(input, axis=expand)


class FeatureBank():
    def __init__(self, num_classes, maxlen=128, disable=False, gamma=1.):
        self.bank = [jnp.array([]) for _ in range(num_classes)]
        self.len = [0 for _ in range(num_classes)]
        self.num_classes = num_classes
        self.maxlen = maxlen
        self.cached = None
        self.disable = disable
        self.gamma = gamma

    def _squeeze(self, batch):
        assert len(batch["images"].shape) == 5 or len(
            batch["images"].shape) == 3
        batch_unpack = dict()
        shapes = dict()
        for k, v in batch.items():
            shapes[k] = v.shape
            batch_unpack[k] = v.reshape(-1, *v.shape[2:])

        return batch_unpack, shapes

    def _unsqueeze(self, batch, shapes):
        assert len(batch["images"].shape) == 4 or len(
            batch["images"].shape) == 2

        new_batch = dict()
        for k, v in batch:
            new_batch[k] = v.reshape(*shapes[k])

        return new_batch

    def deposit(self, batch):
        if self.disable:
            return None
        batch, _ = self._squeeze(batch)
        self._deposit(batch)

    def _deposit(self, batch):
        f_b = batch["images"]
        f_a = batch["labels"]
        labels = batch["cls_labels"]
        marker = batch["marker"]
        f = jnp.concatenate([f_b, f_a], axis=-1)
        shape = f.shape
        f = f[marker, ...]
        if self.cached is None:  # denotes right after bank init
            self.bank = list(
                map(lambda x: x.reshape(-1, *shape[1:]), self.bank))

        def store_fn(i):
            in_class = f[labels == i, ...]
            length = len(in_class)
            exceed = len(self.bank[i]) + length - self.maxlen
            if exceed > 0:
                self.bank[i] = self.bank[i][exceed:]
            self.bank[i] = jnp.concatenate([self.bank[i], in_class], axis=0)
            self.len[i] = len(self.bank[i])
            return val
        val = map(store_fn, range(self.num_classes))
        min_len = min(self.len)

        def trunc_fn(x):
            return x[-min_len:]
        cached = list(map(trunc_fn, self.bank))
        self.cached = jnp.stack(cached)

    def withdraw(self, rng, _batch):
        if self.disable:
            return _batch
        batch, shapes = self._squeeze(_batch)
        labels = batch["cls_labels"]
        out = self._withdraw(rng, labels)
        if out is None:
            return _batch
        assert out.shape == batch["images"].shape
        f_b, f_a = jnp.split(out, 2, axis=-1)
        marker = batch["marker"]
        batch["images"] = batch["images"].at[marker, ...].set(f_b)
        batch["labels"] = batch["labels"].at[marker, ...].set(f_a)
        batch = self._unsqueeze(batch, shapes)
        return batch

    def _withdraw(self, rng, labels):
        min_len = min(self.len)
        if min_len == 0:
            return None
        indices = jax.random.randint(rng, (len(labels),), 0, min_len)
        new = self.cached[labels, indices]
        return new

    def mixup_inclass(self, rng, batch, alpha=1.0):
        if self.disable:
            return batch
        # rng = jax_utils.unreplicate(rng)

        f_b = batch["images"]
        f_a = batch["labels"]
        self.deposit(batch)

        beta_rng, perm_rng = jax.random.split(rng)
        lam = jnp.where(alpha > 0, jax.random.beta(beta_rng, alpha, alpha), 1)
        lam *= self.gamma
        ingredient_batch = self.withdraw(perm_rng, batch)
        ing_b = ingredient_batch["images"]
        ing_a = ingredient_batch["labels"]

        mixed_b = lam*f_b + (1-lam)*ing_b
        mixed_a = lam*f_a + (1-lam)*ing_a
        batch["images"] = mixed_b
        batch["labels"] = mixed_a
        return batch

    def perm_aug(self, rng, batch):
        n_cls = batch["images"].shape[-1]
        labels = batch["cls_labels"]
        ps, bs = labels.shape
        assert len(labels.shape) == 2
        order = jnp.tile(jnp.arange(0, n_cls-1)[None, None, :], [ps, bs, 1])
        order = jnp.where(order >= labels[..., None], order+1, order)
        perm_order = jax.random.permutation(rng, order, axis=-1)
        perm = perm_order.reshape(-1, *perm_order.shape[2:])
        l = labels.reshape(-1, *labels.shape[2:])
        perm_order = jax.vmap(jnp.insert, (0, 0, 0), 0)(perm, l, l)

        def mixer(value):
            assert len(value.shape) > 2
            shape = value.shape
            value = value.reshape(-1, *shape[2:])
            mix = jax.vmap(lambda v, p: v[..., p], (0, 0), 0)
            result = mix(value, perm_order)
            return result.reshape(*shape)

        for k, v in batch.items():
            if v.shape[-1] == n_cls:
                batch[k] = mixer(v)
            if k == "cls_labels":
                labels = common_utils.onehot(v, n_cls)
                batch[k] = jnp.argmax(mixer(labels), axis=-1)

        return batch


def evaluate_top2acc(confidences, true_labels, log_input=True, eps=1e-8, reduction="mean"):

    pred_labels = jnp.argmax(confidences, axis=1)
    mask = common_utils.onehot(pred_labels, confidences.shape[-1])
    temp = -mask*1e10+(1-mask)*confidences
    pred2_labels = jnp.argmax(temp, axis=1)
    raw_results = jnp.equal(pred_labels, true_labels)
    raw2_results = jnp.equal(pred2_labels, true_labels)
    raw_results = jnp.logical_or(raw_results, raw2_results)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


def evaluate_topNacc(confidences, true_labels, top=5, log_input=True, eps=1e-8, reduction="mean"):

    pred_labels = jnp.argmax(confidences, axis=1)
    mask = common_utils.onehot(pred_labels, confidences.shape[-1])
    temp = -mask*1e10+(1-mask)*confidences
    raw_results = jnp.equal(pred_labels, true_labels)
    for i in range(1, top):
        pred2_labels = jnp.argmax(temp, axis=1)
        mask2 = common_utils.onehot(pred2_labels, confidences.shape[-1])
        temp = -mask2*1e10+(1-mask2)*temp
        raw2_results = jnp.equal(pred2_labels, true_labels)
        raw_results = jnp.logical_or(raw_results, raw2_results)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')
