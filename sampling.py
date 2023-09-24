# generic packages

# jax-related packages
import jax
import jax.numpy as jnp

# user-defined packages
from utils import batch_mul

def dsb_sample(score, rng, x0, y0=None, config=None, dsb_stats=None, z_dsb_stats=None, steps=None):
    shape = x0.shape
    batch_size = shape[0]
    _sigma_t = dsb_stats["sigma_t"]
    _alpos_weight_t = dsb_stats["alpos_weight_t"]
    _sigma_t_square = dsb_stats["sigma_t_square"]
    n_T = dsb_stats["n_T"]
    _t = jnp.array([1/n_T])
    _t = jnp.tile(_t, [batch_size])
    std_arr = jnp.sqrt(_alpos_weight_t*_sigma_t_square)
    h_arr = jax.random.normal(rng, (len(_sigma_t), *shape))
    h_arr = h_arr.at[0].set(0)

    @jax.jit
    def body_fn(n, val):
        x_n = val
        idx = n_T - n
        t_n = idx * _t

        h = h_arr[idx]  # (B, d)
        if config.mimo_cond:
            t_n = jnp.tile(t_n[:, None], reps=[1, config.fat])
        eps = score(x_n, y0, t=t_n)

        sigma_t = _sigma_t[idx]
        alpos_weight_t = _alpos_weight_t[idx]
        std = std_arr[idx]

        x_0_eps = x_n - sigma_t*eps
        mean = alpos_weight_t*x_0_eps + (1-alpos_weight_t)*x_n
        x_n = mean + std * h  # (B, d)

        return x_n

    x_list = [x0]
    val = x0
    if steps is None:
        steps = n_T
    for i in range(0, steps):
        val = body_fn(i, val)
        x_list.append(val)
    x_n = val

    return jnp.concatenate(x_list, axis=0)


def dsb_sample_cont(score, rng, x0, y0=None, config=None, dsb_stats=None, z_dsb_stats=None, steps=None):
    shape = x0.shape
    batch_size = shape[0]

    timesteps = jnp.concatenate(
        [jnp.linspace(1.0, 0.001, steps), jnp.zeros([1])])

    @jax.jit
    def body_fn(n, val):
        rng, x_n, x_list = val
        t, next_t = timesteps[n], timesteps[n+1]
        vec_t = jnp.ones([batch_size]) * t
        vec_next_t = jnp.ones([batch_size]) * next_t

        if config.mimo_cond:
            vec_t = jnp.tile(vec_t[:, None], reps=[1, config.fat])
        eps = score(x_n, y0, t=vec_t)

        coeffs = dsb_stats((vec_next_t, vec_t), mode='sampling')
        x_0_eps = x_n - batch_mul(coeffs['sigma_t'], eps)
        mean = batch_mul(coeffs['x0'], x_0_eps) + batch_mul(coeffs['x1'], x_n)

        rng, step_rng = jax.random.split(rng)
        h = jax.random.normal(step_rng, x_n.shape)
        x_n = mean + batch_mul(coeffs['noise'], h)

        x_list.pop(0)
        x_list.append(x_n)
        return rng, x_n, x_list

    x_list = [x0] * (steps + 1)
    # _, _, x_list = jax.lax.fori_loop(0, steps, body_fn, (rng, x0, x_list))
    val = (rng, x0, x_list)
    for i in range(0, steps):
        val = body_fn(i, val)
    _, _, x_list = val
    return jnp.concatenate(x_list, axis=0)

