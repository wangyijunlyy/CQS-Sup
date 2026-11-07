"""NC experimental h_qs on expanded gate coordinates (public-code profile).

The original random sampler is executed unchanged. The JAX solver compiles the
same per-output Adam/EMA loop; independent stopping decisions are retained.
This adapts the algorithm, not the original physical drive, to cached tasks.
"""
from pathlib import Path
from functools import partial, lru_cache
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
import optax
from nc_reference_fixture import load_definitions

jax.config.update('jax_enable_x64', True)


@lru_cache(None)
def reference_namespace():
    root = Path(__file__).resolve().parents[2]
    ns = dict(jax=jax, jnp=jnp, np=np, optax=optax, Optional=Optional)
    load_definitions(root/'tmp/nc_predictive_surrogates_upstream/lib/surrogate.py',
                     ['get_w_vectors_random', 'truncated_trigonometric_features', 'Ridgejax'], ns)
    return ns


def feature_counts(w, mapping, dimensions):
    counts = np.zeros((len(w), dimensions, 2), dtype=np.int32)
    for gate, control in enumerate(mapping):
        counts[:, control, 0] += w[:, gate] == 1
        counts[:, control, 1] += w[:, gate] == -1
    return counts


def sample_counts(n, periods, count=1000, support=25):
    ns = reference_namespace()
    mapping = np.tile([0]*n+list(range(1, n+1)), periods)
    w = np.asarray(ns['get_w_vectors_random'](len(mapping), support, count, jax.random.key(1234)))
    return feature_counts(w, mapping, n+1)


def features(x, counts):
    x = np.asarray(x, dtype=np.float64)
    result = np.ones((len(x), len(counts)), dtype=np.float64)
    for d in range(x.shape[1]):
        result *= np.cos(x[:, d, None])**counts[None, :, d, 0]
        result *= np.sin(x[:, d, None])**counts[None, :, d, 1]
    return result


@partial(jax.jit, static_argnames=('lr', 'epochs', 'threshold'))
def _fit(phi, targets, alpha, lr, epochs, threshold):
    opt = optax.adam(lr)
    initial = jax.random.uniform(jax.random.key(1234), (phi.shape[1],), minval=-.1, maxval=.1)

    def single(y):
        def objective(theta):
            return jnp.sum(jnp.square(phi@theta-y))+alpha*jnp.linalg.norm(theta)

        def body(state):
            i, theta, opt_state, avg, stopped = state
            loss, grad = jax.value_and_grad(objective)(theta)
            new_avg = .9*avg+.1*loss
            stop = (i > 0) & (jnp.abs(new_avg-avg)/(jnp.abs(avg)+1e-8) < threshold)
            updates, new_state = opt.update(grad, opt_state, theta)
            next_theta = jnp.where(stop, theta, optax.apply_updates(theta, updates))
            return i+1, next_theta, new_state, new_avg, stop

        state = jax.lax.while_loop(lambda s: (s[0]<epochs) & ~s[4], body,
                                   (jnp.array(0), initial, opt.init(initial), jnp.array(0.), jnp.array(False)))
        return state[1], state[0]-state[4].astype(jnp.int32)

    return jax.vmap(single, in_axes=1, out_axes=(1, 0))(targets)


def fit_heads(phi, targets, alpha=1., lr=.1, epochs=10000, threshold=1e-5):
    coef, steps = _fit(jnp.asarray(phi, dtype=jnp.float64), jnp.asarray(targets, dtype=jnp.float64),
                       float(alpha), float(lr), int(epochs), float(threshold))
    return np.asarray(coef), np.asarray(steps)
