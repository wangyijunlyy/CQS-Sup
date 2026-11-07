"""Execute selected public-code definitions unchanged, excluding module side effects.

Run with JAX 0.4.38/Optax 0.2.4 in an isolated environment. The complete upstream
MIT source is retained separately; AST selection only avoids unrelated imports,
hard-coded author paths, and top-level experiment execution.
"""
import argparse
import ast
import contextlib
import io
from pathlib import Path
from typing import Optional
import jax
import jax.numpy as jnp
import optax
import numpy as np

jax.config.update("jax_enable_x64", True)


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError("Reference definitions missing")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--widths", type=int, nargs="*", default=[])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for width in args.widths:
        initial = jax.random.uniform(jax.random.key(1234), (width,), minval=-.1, maxval=.1)
        np.save(args.output/f"initial_f{width}.npy", np.asarray(initial))
    namespace = dict(jax=jax, jnp=jnp, np=np, optax=optax, Optional=Optional)
    load_definitions(args.upstream/"lib/surrogate.py",
        ["generate_vectors_l0", "get_w_vectors_l0_jax"], namespace)
    load_definitions(args.upstream/"module/model.py",
        ["truncated_trigonometric_features", "Ridgejax"], namespace)
    w = namespace["get_w_vectors_l0_jax"](4, 2)
    x = np.random.default_rng(19).uniform(-np.pi, np.pi, (43, 4))
    y = np.sin(x[:, 0]) + .4*np.cos(x[:, 1])*np.sin(x[:, 2])
    f = namespace["truncated_trigonometric_features"](jnp.array(x), w)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        model = namespace["Ridgejax"](alpha=1.).fit(f, jnp.array(y), lr=.01, epochs=5000)
    updates = sum(line.startswith("fitting, Step") for line in log.getvalue().splitlines())
    initial = jax.random.uniform(jax.random.key(1234), (len(w),), minval=-.1, maxval=.1)
    np.savez(args.output/"jax_reference.npz", w=np.asarray(w), x=x, y=y,
             features=np.asarray(f), initial=np.asarray(initial), coef=np.asarray(model.theta_),
             updates=updates)
    print(f"Original JAX reference completed: {len(w)} features, {updates} Adam updates")
