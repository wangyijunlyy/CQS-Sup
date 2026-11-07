"""
Quantum circuit parameter pretraining and surrogate modeling using classical shadows.
Implements ridge regression with truncated trigonometric features for efficient
quantum dynamics prediction and circuit initialization.
"""

import sys
from typing import Callable, List, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
from functools import partial
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Local module imports
sys.path.append('/home/lwy/shadow')
from lib.utils import caculate_shadow_expvals
from pennylane import ClassicalShadow


def generate_vectors_l0(length: int, max_nonzero: int, current_vector: Optional[np.ndarray] = None,
                       current_index: int = 0, nonzero_count: int = 0):
    """
    Recursively generate all sparse vectors with elements in {-1, 0, 1} and at most max_nonzero non-zero entries.
    
    Args:
        length: Vector length
        max_nonzero: Maximum number of non-zero elements allowed
        current_vector: Current partial vector during recursion
        current_index: Current index being processed
        nonzero_count: Current count of non-zero elements
        
    Yields:
        Sparse vectors as numpy arrays
    """
    if current_vector is None:
        current_vector = np.zeros(length, dtype=int)
    
    if current_index == length:
        if nonzero_count <= max_nonzero:
            yield current_vector.copy()
        return
    
    for value in [-1, 0, 1]:
        new_nonzero_count = nonzero_count + (value != 0)
        if new_nonzero_count <= max_nonzero:
            current_vector[current_index] = value
            yield from generate_vectors_l0(
                length, max_nonzero, current_vector, current_index + 1, new_nonzero_count
            )


def get_w_vectors_l0_jax(length: int, max_nonzero: int) -> jnp.ndarray:
    """
    Generate all sparse weight vectors with L0 constraint using JAX.
    
    Args:
        length: Vector length (number of parameters)
        max_nonzero: Maximum number of non-zero elements (L0 constraint)
        
    Returns:
        JAX array of shape (num_vectors, length) containing all valid sparse vectors
    """
    return jnp.array(list(generate_vectors_l0(length, max_nonzero)))


def get_w_vectors_random(length: int, max_nonzero: int, nf: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Randomly generate nf sparse vectors with elements in {-1, 0, 1} and at most max_nonzero non-zero entries.
    
    Args:
        length: Vector length
        max_nonzero: Maximum number of non-zero elements allowed
        nf: Number of vectors to generate
        key: JAX random key
        
    Returns:
        JAX array of shape (nf, length) containing random sparse vectors
    """
    max_nonzero = min(max_nonzero, length)
    if max_nonzero == 0:
        return jnp.zeros((nf, length), dtype=jnp.int32)
    
    key_k, key_pos, key_sign = jax.random.split(key, 3)
    k_vals = jax.random.randint(key_k, (nf,), 0, max_nonzero + 1)
    
    keys_pos = jax.random.split(key_pos, nf)
    
    def gen_pos(k, size, length_val):
        return jax.random.choice(k, length_val, shape=(size,), replace=False)
    
    indices = jax.vmap(gen_pos, in_axes=(0, None, None))(
        keys_pos, max_nonzero, length
    )
    
    signs = jax.random.choice(
        key_sign, jnp.array([-1, 1]), shape=(nf, max_nonzero)
    )
    
    vectors = jnp.zeros((nf, length), dtype=jnp.int32)
    
    def set_values(i, vecs):
        k = k_vals[i]
        if k > 0:
            pos = indices[i, :k]
            vals = signs[i, :k]
            vecs = vecs.at[i, pos].set(vals)
        return vecs
    
    for i in range(nf):
        vectors = set_values(i, vectors)
    
    return vectors


@jax.jit
def truncated_trigonometric_features(X_train: jnp.ndarray, W: jnp.ndarray) -> jnp.ndarray:
    """
    Compute truncated trigonometric features for parameterized quantum circuits.
    
    For each parameter x_i and weight w_i ∈ {-1, 0, 1}:
      - w_i = 1  → cos(x_i)
      - w_i = -1 → sin(x_i)
      - w_i = 0  → 1
    
    Args:
        X_train: Input parameters of shape (n_samples, n_params)
        W: Weight vectors of shape (n_features, n_params)
        
    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    cos_cds_nsample = jnp.cos(X_train)
    sin_cds_nsample = jnp.sin(X_train)
    
    def x_map(w_term, cos_cd, sin_cd):
        return jnp.where(w_term == 1, cos_cd, jnp.where(w_term == -1, sin_cd, 1))
    
    def phix_element(w, cos_cds, sin_cds):
        return jnp.prod(jax.vmap(x_map, in_axes=(0, 0, 0))(w, cos_cds, sin_cds))
    
    def phix(wt, cos_cds, sin_cds):
        return jax.vmap(phix_element, in_axes=(0, None, None))(wt, cos_cds, sin_cds)
    
    return jax.vmap(phix, in_axes=(None, 0, 0))(W, cos_cds_nsample, sin_cds_nsample)


class Ridgejax:
    """Ridge regression implementation using JAX with early stopping support."""
    
    def __init__(self, alpha: float = 10.0, theta: Optional[jnp.ndarray] = None):
        """
        Initialize Ridge regression model.
        
        Args:
            alpha: Regularization strength (default: 10.0)
            theta: Pre-trained parameters (optional)
        """
        self.alpha = alpha
        self.theta_ = theta
    
    def predict(self, test_feature: jnp.ndarray) -> jnp.ndarray:
        """
        Predict target values for test features.
        
        Args:
            test_feature: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
        """
        if self.theta_ is None:
            raise ValueError('Model must be trained before prediction')
        return jnp.matmul(self.theta_, test_feature.T)
    
    def fit(
        self,
        features: jnp.ndarray,
        y: jnp.ndarray,
        lr: float = 0.01,
        epochs: int = 5000,
        seed: int = 1234,
        ep: bool = True,
        stop_threshold: float = 1e-4,
        display: bool = True
    ) -> "Ridgejax":
        """
        Fit ridge regression model using Adam optimizer with optional early stopping.
        
        Args:
            features: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            lr: Learning rate (default: 0.01)
            epochs: Maximum training epochs (default: 5000)
            seed: Random seed for parameter initialization (default: 1234)
            ep: Enable early stopping (default: True)
            stop_threshold: Early stopping threshold based on moving average change (default: 1e-4)
            display: Print training progress (default: True)
            
        Returns:
            Self for method chaining
        """
        @jax.jit
        def loss(theta, features_arr, y_arr, alpha_val):
            pred = jnp.matmul(theta, features_arr.T)
            mse = jnp.sum(jnp.square(pred - y_arr))
            reg = alpha_val * jnp.linalg.norm(theta, ord=2)
            return mse + reg
        
        theta = jax.random.uniform(
            jax.random.key(seed), (features.shape[1],), minval=-0.1, maxval=0.1
        )
        opt = optax.adam(learning_rate=lr)
        opt_state = opt.init(theta)
        value_and_grad_fn = jax.jit(jax.value_and_grad(loss))
        
        decay = 0.9
        avg_loss = 0.0
        prev_loss = None
        alpha_val = self.alpha
        
        for i in range(epochs):
            loss_val, grads = value_and_grad_fn(theta, features, y, alpha_val)
            avg_loss = decay * avg_loss + (1 - decay) * loss_val
            
            if ep and prev_loss is not None:
                avg_loss_rate_change = jnp.abs(avg_loss - prev_loss) / (jnp.abs(prev_loss) + 1e-8)
                if avg_loss_rate_change < stop_threshold:
                    if display:
                        print(f"Stopping early at step {i} due to small avg loss rate change: "
                              f"{avg_loss_rate_change:.6f} with avg_loss: {avg_loss:.6f}")
                    break
            
            prev_loss = avg_loss
            updates, opt_state = opt.update(grads, opt_state, theta)
            theta = optax.apply_updates(theta, updates)
        
        self.theta_ = theta
        if display:
            print(f'Fitting completed. Final loss: {loss_val:.6f}, Avg loss: {avg_loss:.6f}')
        return self


def evaluate_predictions(predictions: np.ndarray, test_results: np.ndarray) -> dict:
    """
    Evaluate prediction quality using multiple metrics.
    
    Args:
        predictions: Predicted values
        test_results: True values
        
    Returns:
        Dictionary containing R², Pearson correlation, MSE, and MAE
    """
    r2 = r2_score(test_results, predictions)
    pearson_corr, _ = pearsonr(test_results, predictions)
    mse = mean_squared_error(test_results, predictions)
    mae = mean_absolute_error(test_results, predictions)
    return {
        "R^2": r2,
        "Pearson Correlation": pearson_corr,
        "MSE": mse,
        "MAE": mae
    }


class RidgePretrainer:
    """Pretrainer for quantum circuit parameters using classical shadow data."""
    
    def __init__(self, params: jnp.ndarray, shadows: list, H):
        """
        Initialize pretrainer with shadow data and Hamiltonian.
        
        Args:
            params: Circuit parameters of shape (n_samples, n_params)
            shadows: List of ClassicalShadow objects
            H: Pennylane observable representing the Hamiltonian
        """
        self.params = params
        self.shadows = shadows
        self.H = H
        self.n_param = params.shape[1]
        self.Lambda_ = None
        self.trial_params_ = None
        self.trial_shadows_ = None
        self.trial_expvals_ = None
        self.ntest_ = None
        self.epochs_ = None
        self.lr_ = None
        self.x_rng_ = None
        self.w_jnp = None
    
    def set(
        self,
        ushadow: int,
        usnapshot: int,
        Lambda: int,
        ntest: int,
        epochs: int,
        lr: float,
        x_rng: int = 1234
    ) -> "RidgePretrainer":
        """
        Configure pretraining hyperparameters.
        
        Args:
            ushadow: Number of shadow samples to use
            usnapshot: Number of snapshots per shadow
            Lambda: L0 sparsity constraint for feature vectors
            ntest: Number of independent optimization trials
            epochs: Maximum optimization steps per trial
            lr: Learning rate for parameter updates
            x_rng: Random seed for parameter initialization
            
        Returns:
            Self for method chaining
        """
        self.ushadow_ = ushadow
        all_bits = np.array([shadow.bits for shadow in self.shadows])
        all_recipes = np.array([shadow.recipes for shadow in self.shadows])
        bits = all_bits[-ushadow:, -usnapshot:, :]
        recipes = all_recipes[-ushadow:, -usnapshot:, :]
        
        trial_shadows_ = []
        for bit, recipe in zip(bits, recipes):
            bit = bit.astype('float64')
            trial_shadows_.append(ClassicalShadow(bit, recipe))
        
        self.trial_params_ = self.params[-ushadow:]
        self.trial_shadows_ = trial_shadows_
        self.trial_expvals_ = caculate_shadow_expvals(self.trial_shadows_, self.H)
        self.Lambda_ = Lambda
        self.ntest_ = ntest
        self.epochs_ = epochs
        self.lr_ = lr
        self.x_rng_ = x_rng
        self.w_jnp = get_w_vectors_l0_jax(self.n_param, self.Lambda_)
        return self
    
    def execute(
        self,
        circuit: Optional[Callable] = None,
        display: bool = True
    ) -> Tuple[List[List[jnp.ndarray]], List[List[float]], List[List[Optional[float]]]]:
        """
        Execute pretraining optimization across multiple random initializations.
        
        Args:
            circuit: Optional quantum circuit for true energy evaluation
            display: Print training progress
            
        Returns:
            Tuple containing:
                - Parameter trajectories for each trial
                - Loss histories for each trial
                - True energy evaluations for each trial (if circuit provided)
        """
        model = Ridgejax()
        model_fitted = model.fit(
            features=truncated_trigonometric_features(self.trial_params_, self.w_jnp),
            y=self.trial_expvals_,
            lr=0.01,
            epochs=5000,
            display=False
        )
        
        def pretrain(key: jax.random.PRNGKey):
            x = jax.random.uniform(key, (self.n_param,), minval=-jnp.pi, maxval=jnp.pi)
            loss_history = []
            x_history = []
            true_history = []
            
            opt = optax.adam(learning_rate=self.lr_)
            opt_state = opt.init(x)
            
            @jax.jit
            def loss_fn(x_param):
                x_reshaped = jnp.reshape(x_param, (1, x_param.shape[0]))
                features = truncated_trigonometric_features(x_reshaped, self.w_jnp)
                return jnp.mean(model_fitted.predict(features))
            
            value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
            decay = 0.95
            avg_loss = 0.0
            stop_threshold = 1e-5
            prev_loss = None
            
            for i in range(self.epochs_):
                loss_val, grads = value_and_grad_fn(x)
                true_val = circuit(x) if circuit is not None else None
                
                avg_loss = decay * avg_loss + (1 - decay) * loss_val
                loss_history.append(loss_val)
                x_history.append(x.copy())
                true_history.append(true_val)
                
                if prev_loss is not None:
                    avg_loss_rate_change = jnp.abs(avg_loss - prev_loss) / (jnp.abs(prev_loss) + 1e-8)
                    if avg_loss_rate_change < stop_threshold:
                        break
                
                prev_loss = avg_loss
                updates, opt_state = opt.update(grads, opt_state, x)
                x = optax.apply_updates(x, updates)
                x = jax.tree.map(lambda v: jnp.clip(v, -jnp.pi, jnp.pi), x)
                
                if display:
                    print(f'Step {i}: Loss {loss_val:.6f}, Avg Loss {avg_loss:.6f}, True {true_val}')
            
            return x_history, loss_history, true_history
        
        keys = jax.random.split(jax.random.key(self.x_rng_), self.ntest_)
        all_x_histories = []
        all_loss_histories = []
        all_true_histories = []
        
        for key in keys:
            x_history, loss_history, true_history = pretrain(key)
            all_x_histories.append(x_history)
            all_loss_histories.append(loss_history)
            all_true_histories.append(true_history)
        
        return all_x_histories, all_loss_histories, all_true_histories


class RidgeExpvalPretrainer:
    """Pretrainer using precomputed expectation values (no shadow data required)."""
    
    def __init__(self, training_params: jnp.ndarray, training_labels: jnp.ndarray):
        """
        Initialize pretrainer with precomputed expectation values.
        
        Args:
            training_params: Circuit parameters of shape (n_samples, n_params)
            training_labels: Corresponding expectation values of shape (n_samples,)
        """
        self.training_params = training_params
        self.training_labels = training_labels
        self.n_param = training_params.shape[1]
        self.Lambda_ = None
        self.trial_training_params_ = None
        self.trial_training_labels_ = None
        self.ntest_ = None
        self.pretrain_epochs_ = None
        self.fit_epochs_ = None
        self.pretrain_lr_ = None
        self.fit_lr_ = None
        self.alpha_ = None
        self.x_rng_ = None
        self.w_jnp = None
    
    def set(
        self,
        usample: int,
        Lambda: int,
        ntest: int,
        fit_epochs: int = 5000,
        fit_lr: float = 0.01,
        pretrain_epochs: int = 5000,
        pretrain_lr: float = 0.01,
        alpha: float = 1000.0,
        x_rng: int = 1234
    ) -> "RidgeExpvalPretrainer":
        """
        Configure pretraining hyperparameters.
        
        Args:
            usample: Number of samples to use for training
            Lambda: L0 sparsity constraint
            ntest: Number of independent optimization trials
            fit_epochs: Epochs for ridge regression fitting
            fit_lr: Learning rate for ridge regression
            pretrain_epochs: Epochs for parameter pretraining
            pretrain_lr: Learning rate for parameter pretraining
            alpha: Ridge regularization strength
            x_rng: Random seed
            
        Returns:
            Self for method chaining
        """
        self.usample_ = usample
        self.trial_training_params_ = self.training_params[-usample:]
        self.trial_training_labels_ = self.training_labels[-usample:]
        self.Lambda_ = Lambda
        self.ntest_ = ntest
        self.pretrain_epochs_ = pretrain_epochs
        self.fit_epochs_ = fit_epochs
        self.pretrain_lr_ = pretrain_lr
        self.fit_lr_ = fit_lr
        self.alpha_ = alpha
        self.x_rng_ = jax.random.key(x_rng)
        self.w_jnp = get_w_vectors_l0_jax(self.n_param, self.Lambda_)
        return self
    
    def execute(
        self,
        circuit: Optional[Callable] = None,
        display: bool = True
    ) -> Tuple[List[List[jnp.ndarray]], List[List[float]], List[List[Optional[float]]]]:
        """
        Execute pretraining optimization.
        
        Args:
            circuit: Optional quantum circuit for true energy evaluation
            display: Print training progress
            
        Returns:
            Tuple containing parameter trajectories, loss histories, and true energy evaluations
        """
        model = Ridgejax(alpha=self.alpha_)
        model_fitted = model.fit(
            features=truncated_trigonometric_features(
                self.trial_training_params_, self.w_jnp
            ),
            y=self.trial_training_labels_,
            lr=self.fit_lr_,
            epochs=self.fit_epochs_,
            display=False
        )
        
        def pretrain(key: jax.random.PRNGKey):
            x = jax.random.uniform(key, (self.n_param,), minval=-jnp.pi, maxval=jnp.pi)
            loss_history = []
            x_history = []
            true_history = []
            
            opt = optax.adam(learning_rate=self.pretrain_lr_)
            opt_state = opt.init(x)
            
            @jax.jit
            def loss_fn(x_param):
                x_reshaped = jnp.reshape(x_param, (1, x_param.shape[0]))
                features = truncated_trigonometric_features(x_reshaped, self.w_jnp)
                return jnp.mean(model_fitted.predict(features))
            
            value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
            decay = 0.95
            avg_loss = 0.0
            stop_threshold = 1e-4
            prev_loss = None
            
            for i in range(self.pretrain_epochs_):
                loss_val, grads = value_and_grad_fn(x)
                true_val = circuit(x) if circuit is not None else None
                
                avg_loss = decay * avg_loss + (1 - decay) * loss_val
                loss_history.append(loss_val)
                x_history.append(x.copy())
                true_history.append(true_val)
                
                if prev_loss is not None:
                    avg_loss_rate_change = jnp.abs(avg_loss - prev_loss) / (jnp.abs(prev_loss) + 1e-8)
                    if avg_loss_rate_change < stop_threshold:
                        break
                
                prev_loss = avg_loss
                updates, opt_state = opt.update(grads, opt_state, x)
                x = optax.apply_updates(x, updates)
                x = jax.tree.map(lambda v: jnp.clip(v, -jnp.pi, jnp.pi), x)
                
                if display:
                    print(f'Step {i}: Loss {loss_val:.6f}, Avg Loss {avg_loss:.6f}, True {true_val}')
            
            return x_history, loss_history, true_history
        
        keys = jax.random.split(self.x_rng_, self.ntest_)
        all_x_histories = []
        all_loss_histories = []
        all_true_histories = []
        
        for key in keys:
            x_history, loss_history, true_history = pretrain(key)
            all_x_histories.append(x_history)
            all_loss_histories.append(loss_history)
            all_true_histories.append(true_history)
        
        return all_x_histories, all_loss_histories, all_true_histories


class ShareRidgeExpvalPretrainer:
    """Shared-parameter pretrainer for translation-invariant quantum circuits."""
    
    def __init__(self, training_params: jnp.ndarray, training_labels: jnp.ndarray):
        """
        Initialize shared-parameter pretrainer.
        
        Args:
            training_params: Circuit parameters of shape (n_samples, n_params)
            training_labels: Corresponding expectation values of shape (n_samples,)
        """
        self.training_params = training_params
        self.training_labels = training_labels
        self.n_param = training_params.shape[1]
        self.Lambda_ = None
        self.trial_training_params_ = None
        self.trial_training_labels_ = None
        self.ntest_ = None
        self.pretrain_epochs_ = None
        self.fit_epochs_ = None
        self.pretrain_lr_ = None
        self.fit_lr_ = None
        self.alpha_ = None
        self.x_rng_ = None
        self.w_jnp = None
    
    def set(
        self,
        usample: int,
        Lambda: int,
        ntest: int,
        fit_epochs: int = 5000,
        fit_lr: float = 0.01,
        pretrain_epochs: int = 5000,
        pretrain_lr: float = 0.01,
        alpha: float = 1000.0,
        x_rng: int = 1234
    ) -> "ShareRidgeExpvalPretrainer":
        """
        Configure pretraining hyperparameters.
        
        Args:
            usample: Number of samples to use
            Lambda: L0 sparsity constraint
            ntest: Number of optimization trials
            fit_epochs: Ridge regression epochs
            fit_lr: Ridge regression learning rate
            pretrain_epochs: Parameter pretraining epochs
            pretrain_lr: Parameter pretraining learning rate
            alpha: Ridge regularization strength
            x_rng: Random seed
            
        Returns:
            Self for method chaining
        """
        self.usample_ = usample
        self.trial_training_params_ = self.training_params[-usample:]
        self.trial_training_labels_ = self.training_labels[-usample:]
        self.Lambda_ = Lambda
        self.ntest_ = ntest
        self.pretrain_epochs_ = pretrain_epochs
        self.fit_epochs_ = fit_epochs
        self.pretrain_lr_ = pretrain_lr
        self.fit_lr_ = fit_lr
        self.alpha_ = alpha
        self.x_rng_ = jax.random.key(x_rng)
        self.w_jnp = get_w_vectors_l0_jax(self.n_param, self.Lambda_)
        return self
    
    def execute(
        self,
        n_qubit: int,
        feature_map: Callable,
        circuit: Optional[Callable] = None,
        display: bool = True
    ) -> Tuple[List[List[jnp.ndarray]], List[List[float]], List[List[Optional[float]]]]:
        """
        Execute shared-parameter pretraining.
        
        Args:
            n_qubit: Number of physical qubits
            feature_map: Feature mapping function for shared parameters
            circuit: Optional quantum circuit for true energy evaluation
            display: Print training progress
            
        Returns:
            Tuple containing parameter trajectories, loss histories, and true energy evaluations
        """
        model = Ridgejax(alpha=self.alpha_)
        model_fitted = model.fit(
            features=feature_map(self.trial_training_params_, self.w_jnp),
            y=self.trial_training_labels_,
            lr=self.fit_lr_,
            epochs=self.fit_epochs_,
            display=False
        )
        
        def pretrain(key: jax.random.PRNGKey):
            # Shared parameters: [theta1, theta2] for all qubits
            x = jax.random.uniform(key, (2,), minval=-jnp.pi, maxval=jnp.pi)
            loss_history = []
            x_history = []
            true_history = []
            
            opt = optax.adam(learning_rate=self.pretrain_lr_)
            opt_state = opt.init(x)
            
            def loss_fn(x_param):
                # Unpack shared parameters to full circuit parameters
                x_unpacked = jnp.repeat(x_param, jnp.array([n_qubit, self.n_param - n_qubit]))
                x_reshaped = jnp.reshape(x_unpacked, (1, x_unpacked.shape[0]))
                features = feature_map(x_reshaped, self.w_jnp)
                return jnp.mean(model_fitted.predict(features))
            
            value_and_grad_fn = jax.value_and_grad(loss_fn)
            decay = 0.95
            avg_loss = 0.0
            stop_threshold = 1e-4
            prev_loss = None
            
            for i in range(self.pretrain_epochs_):
                loss_val, grads = value_and_grad_fn(x)
                true_val = circuit(x) if circuit is not None else None
                
                avg_loss = decay * avg_loss + (1 - decay) * loss_val
                loss_history.append(loss_val)
                x_history.append(x.copy())
                true_history.append(true_val)
                
                if prev_loss is not None:
                    avg_loss_rate_change = jnp.abs(avg_loss - prev_loss) / (jnp.abs(prev_loss) + 1e-8)
                    if avg_loss_rate_change < stop_threshold:
                        break
                
                prev_loss = avg_loss
                updates, opt_state = opt.update(grads, opt_state, x)
                x = optax.apply_updates(x, updates)
                
                if display:
                    print(f'Step {i}: Loss {loss_val:.6f}, Avg Loss {avg_loss:.6f}, True {true_val}')
            
            return x_history, loss_history, true_history
        
        keys = jax.random.split(self.x_rng_, self.ntest_)
        all_losses = []
        all_x_histories = []
        all_loss_histories = []
        all_true_histories = []
        
        for key in keys:
            x_history, loss_history, true_history = pretrain(key)
            all_losses.append(loss_history[-1])
            all_x_histories.append(x_history)
            all_loss_histories.append(loss_history)
            all_true_histories.append(true_history)
        
        # Select top-performing trials
        min_loss = min(all_losses)
        threshold = min_loss * 0.8  # Keep trials within 80% of best loss
        
        valid_x_histories = []
        valid_loss_histories = []
        valid_true_histories = []
        
        for i in range(self.ntest_):
            if all_losses[i] <= threshold:
                valid_x_histories.append(all_x_histories[i])
                valid_loss_histories.append(all_loss_histories[i])
                valid_true_histories.append(all_true_histories[i])
        
        # Generate additional trials if needed
        while len(valid_x_histories) < self.ntest_:
            self.x_rng_, key = jax.random.split(self.x_rng_)
            x_history, loss_history, true_history = pretrain(key)
            if loss_history[-1] <= threshold:
                valid_x_histories.append(x_history)
                valid_loss_histories.append(loss_history)
                valid_true_histories.append(true_history)
        
        if display:
            final_losses = [hl[-1] for hl in valid_loss_histories]
            print(f"Final losses of valid samples: {final_losses}")
        
        return valid_x_histories, valid_loss_histories, valid_true_histories


@partial(jax.jit, static_argnames=['n_q'])
def share_truncated_trigonometric_features_odd(
    X_train: jnp.ndarray,
    W: jnp.ndarray,
    n_q: int
) -> jnp.ndarray:
    """
    Shared-parameter feature map for odd time steps in DTC evolution.
    
    Args:
        X_train: Input parameters of shape (n_samples, 2) [shared params]
        W: Weight vectors of shape (n_features, n_unpack_params)
        n_q: Number of physical qubits
        
    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    n_unpack_params = W.shape[1]
    t = int(((n_unpack_params - n_q) / (2 * n_q - 2)) * 2 + 1)
    cos_cds_nsample = jnp.cos(X_train)
    sin_cds_nsample = jnp.sin(X_train)
    
    def phi_e(w_vec, coscd, sincd):
        split_indices = jnp.arange(0, n_unpack_params, 2 * n_q - 2)[1:]
        split_arr = jnp.split(w_vec, split_indices)
        split_arr_block = jnp.array(split_arr[:-1]) if len(split_arr) > 1 else jnp.array([])
        split_arr_last = jnp.array([split_arr[-1]])
        
        if split_arr_block.shape[0] != 0:
            w_permutation = jnp.concatenate(
                jnp.concatenate((split_arr_block[:, :n_q], split_arr_last), axis=0), axis=0
            )
            w_coupling = split_arr_block[:, n_q:].transpose(1, 0)
            coupling_cos_weight = jnp.sum(w_coupling == 1, axis=1)
            coupling_sin_weight = jnp.sum(w_coupling == -1, axis=1)
            permutation_cos_weight = jnp.sum(w_permutation == 1).reshape(1,)
            permutation_sin_weight = jnp.sum(w_permutation == -1).reshape(1,)
        else:
            w_permutation = jnp.concatenate(split_arr_last, axis=0)
            coupling_cos_weight = jnp.zeros(shape=(n_q - 2,))
            coupling_sin_weight = jnp.zeros(shape=(n_q - 2,))
            permutation_cos_weight = jnp.sum(w_permutation == 1).reshape(1,)
            permutation_sin_weight = jnp.sum(w_permutation == -1).reshape(1,)
        
        cos_weight = jnp.concatenate((permutation_cos_weight, coupling_cos_weight), axis=0)
        sin_weight = jnp.concatenate((permutation_sin_weight, coupling_sin_weight), axis=0)
        x_map = jnp.power(coscd, cos_weight) * jnp.power(sincd, sin_weight)
        return jnp.prod(x_map)
    
    def phix(wt, cos_cds, sin_cds):
        return jax.vmap(phi_e, in_axes=(0, None, None))(wt, cos_cds, sin_cds)
    
    return jax.vmap(phix, in_axes=(None, 0, 0))(W, cos_cds_nsample, sin_cds_nsample)


@partial(jax.jit, static_argnames=['n_q'])
def share_truncated_trigonometric_features_even(
    X_train: jnp.ndarray,
    W: jnp.ndarray,
    n_q: int
) -> jnp.ndarray:
    """
    Shared-parameter feature map for even time steps in DTC evolution.
    
    Args:
        X_train: Input parameters of shape (n_samples, 2) [shared params]
        W: Weight vectors of shape (n_features, n_unpack_params)
        n_q: Number of physical qubits
        
    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    n_unpack_params = W.shape[1]
    t = int((n_unpack_params / (2 * n_q - 2)) * 2)
    cos_cds_nsample = jnp.cos(X_train)
    sin_cds_nsample = jnp.sin(X_train)
    
    def phi_e(w_vec, coscd, sincd):
        split_indices = jnp.arange(0, n_unpack_params, 2 * n_q - 2)[1:]
        split_arr = jnp.split(w_vec, split_indices)
        split_arr_block = jnp.array(split_arr)
        
        if split_arr_block.shape[0] != 0:
            w_permutation = jnp.concatenate((split_arr_block[:, :n_q]), axis=0)
            w_coupling = split_arr_block[:, n_q:].transpose(1, 0)
            coupling_cos_weight = jnp.sum(w_coupling == 1, axis=1)
            coupling_sin_weight = jnp.sum(w_coupling == -1, axis=1)
            permutation_cos_weight = jnp.sum(w_permutation == 1).reshape(1,)
            permutation_sin_weight = jnp.sum(w_permutation == -1).reshape(1,)
        else:
            coupling_cos_weight = jnp.zeros(shape=(n_q - 2,))
            coupling_sin_weight = jnp.zeros(shape=(n_q - 2,))
            permutation_cos_weight = jnp.zeros(shape=(1,))
            permutation_sin_weight = jnp.zeros(shape=(1,))
        
        cos_weight = jnp.concatenate((permutation_cos_weight, coupling_cos_weight), axis=0)
        sin_weight = jnp.concatenate((permutation_sin_weight, coupling_sin_weight), axis=0)
        x_map = jnp.power(coscd, cos_weight) * jnp.power(sincd, sin_weight)
        return jnp.prod(x_map)
    
    def phix(wt, cos_cds, sin_cds):
        return jax.vmap(phi_e, in_axes=(0, None, None))(wt, cos_cds, sin_cds)
    
    return jax.vmap(phix, in_axes=(None, 0, 0))(W, cos_cds_nsample, sin_cds_nsample)


class DTC_surrogate_share:
    """Surrogate model for discrete time crystal (DTC) quantum dynamics with shared parameters."""
    
    def __init__(self, params: jnp.ndarray, expvals: jnp.ndarray):
        """
        Initialize DTC surrogate model.
        
        Args:
            params: Circuit parameters of shape (n_samples, n_params)
            expvals: Time-resolved expectation values of shape (n_qubits, n_samples, n_time_steps)
        """
        self.params = params
        self.expvals = expvals
        self.n_param = params.shape[1]
        self.n_qubit = expvals.shape[0]
        self.n_T = expvals.shape[2]
        self.n_trial_sample_ = None
        self.trial_params_ = None
        self.trial_expvals_ = None
        self.fitted_model = None
        self.W_list = None
    
    def build(
        self,
        n_trial_sample: int,
        num_feature: int,
        p1: float,
        alpha: float,
        lr: float,
        epochs: int = 5000,
        note: Optional[str] = None,
        ep: bool = True
    ) -> "DTC_surrogate_share":
        """
        Build surrogate model by training ridge regression for each qubit and time step.
        
        Args:
            n_trial_sample: Number of samples to use for training
            num_feature: Number of random features
            p1: Probability for non-zero elements in random feature vectors
            alpha: Ridge regularization strength
            lr: Learning rate for training
            epochs: Training epochs per model
            note: Optional note for saving files
            ep: Enable early stopping
            
        Returns:
            Self for method chaining
        """
        self.n_trial_sample_ = n_trial_sample
        self.trial_params_ = self.params[-n_trial_sample:, :]
        self.trial_expvals_ = self.expvals[:, -n_trial_sample:, :]
        
        def get_random_W(num_feature_val, num_param, p_val, key=jax.random.key(123456)):
            probabilities = jnp.array([p_val, 1 - 2 * p_val, p_val])
            return jax.random.choice(
                key,
                jnp.array([-1, 0, 1]),
                shape=(num_feature_val, num_param),
                replace=True,
                p=probabilities
            )
        
        model_list = [[Ridgejax(alpha=alpha) for _ in range(self.n_T)] for _ in range(self.n_qubit)]
        thetas = np.zeros(shape=(self.n_qubit, self.n_T, num_feature))
        W_list = [[None for _ in range(self.n_T)] for _ in range(self.n_qubit)]
        
        for site in range(self.n_qubit):
            for T in range(self.n_T):
                print(f'Q{site+1}, T{T}')
                if T % 2 == 0:
                    n_unpack_params = (2 * self.n_qubit - 2) * (T // 2)
                    W = get_random_W(num_feature, n_unpack_params, p1)
                    model_list[site][T].fit(
                        features=share_truncated_trigonometric_features_even(
                            self.trial_params_, W, n_q=self.n_qubit
                        ),
                        y=self.trial_expvals_[site, :, T],
                        lr=lr,
                        epochs=epochs,
                        ep=ep,
                        stop_threshold=0.00001,
                        display=False
                    )
                else:
                    n_unpack_params = (2 * self.n_qubit - 2) * ((T - 1) // 2) + self.n_qubit
                    W = get_random_W(num_feature, n_unpack_params, p1)
                    model_list[site][T].fit(
                        features=share_truncated_trigonometric_features_odd(
                            self.trial_params_, W, n_q=self.n_qubit
                        ),
                        y=self.trial_expvals_[site, :, T],
                        lr=lr,
                        epochs=epochs,
                        ep=ep,
                        stop_threshold=0.00001,
                        display=False
                    )
                
                thetas[site][T] = np.array(model_list[site][T].theta_)
                W_list[site][T] = W
        
        if note is not None:
            np.save(
                f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/thetas_{self.n_qubit}q_'
                f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{p1}p_{note}.npy',
                thetas
            )
            with open(
                f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/Ws_{self.n_qubit}q_'
                f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{p1}p_{note}.npy',
                'wb'
            ) as f:
                pickle.dump(W_list, f)
        
        self.fitted_model = model_list
        self.W_list = W_list
        return self
    
    def finetune(
        self,
        n_trial_sample: int,
        thetas: np.ndarray,
        W_list: list,
        site: int,
        t: int,
        truncation: float,
        alpha: float,
        Wseed: int,
        thetaseed: int,
        lr: float,
        note: str,
        epochs: int = 10000,
        ep: bool = True,
        stop_threshold: float = 0.0001
    ) -> None:
        """
        Finetune a specific qubit/time-step model with truncated random features.
        
        Args:
            n_trial_sample: Number of samples to use
            thetas: Current theta parameters array
            W_list: Current weight vectors list
            site: Qubit index to finetune
            t: Time step to finetune
            truncation: Truncation parameter for random features
            alpha: Ridge regularization strength
            Wseed: Random seed for weight generation
            thetaseed: Random seed for parameter initialization
            lr: Learning rate
            note: Note for saving files
            epochs: Training epochs
            ep: Enable early stopping
            stop_threshold: Early stopping threshold
        """
        self.n_trial_sample_ = n_trial_sample
        self.trial_params_ = self.params[-n_trial_sample:, :]
        self.trial_expvals_ = self.expvals[:, -n_trial_sample:, :]
        
        def get_random_W(num_feature_val, num_param, p_val, key):
            probabilities = jnp.array([p_val, 1 - 2 * p_val, p_val])
            return jax.random.choice(
                key,
                jnp.array([-1, 0, 1]),
                shape=(num_feature_val, num_param),
                replace=True,
                p=probabilities
            )
        
        model = Ridgejax(alpha=alpha)
        num_feature = W_list[site][t].shape[0]
        
        if t % 2 == 0:
            n_unpack_params = (2 * self.n_qubit - 2) * (t // 2)
            W = get_random_W(num_feature, n_unpack_params, truncation, jax.random.key(Wseed))
            model.fit(
                features=share_truncated_trigonometric_features_even(
                    self.trial_params_, W, n_q=self.n_qubit
                ),
                y=self.trial_expvals_[site, :, t],
                lr=lr,
                epochs=epochs,
                seed=thetaseed,
                ep=ep,
                stop_threshold=stop_threshold,
                display=False
            )
        else:
            n_unpack_params = (2 * self.n_qubit - 2) * ((t - 1) // 2) + self.n_qubit
            W = get_random_W(num_feature, n_unpack_params, truncation, jax.random.key(Wseed))
            model.fit(
                features=share_truncated_trigonometric_features_odd(
                    self.trial_params_, W, n_q=self.n_qubit
                ),
                y=self.trial_expvals_[site, :, t],
                lr=lr,
                epochs=epochs,
                seed=thetaseed,
                ep=ep,
                stop_threshold=stop_threshold,
                display=False
            )
        
        thetas[site][t] = np.array(model.theta_)
        W_list[site][t] = W
        
        np.save(
            f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/thetas_{self.n_qubit}q_'
            f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}p_{note}.npy',
            thetas
        )
        with open(
            f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/Ws_{self.n_qubit}q_'
            f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}p_{note}.npy',
            'wb'
        ) as f:
            pickle.dump(W_list, f)
    
    def load(
        self,
        thetas: np.ndarray,
        W_list: list,
        alpha: float
    ) -> "DTC_surrogate_share":
        """
        Load pre-trained model parameters.
        
        Args:
            thetas: Theta parameters array of shape (n_qubit, n_T, num_feature)
            W_list: Weight vectors list
            alpha: Ridge regularization strength
            
        Returns:
            Self for method chaining
        """
        model_list = [[Ridgejax(alpha=alpha) for _ in range(self.n_T)] for _ in range(self.n_qubit)]
        for site in range(self.n_qubit):
            for T in range(self.n_T):
                model_list[site][T].theta_ = thetas[site][T]
        
        self.fitted_model = model_list
        self.W_list = W_list
        return self
    
    def predict(self, x: jnp.ndarray, t: int) -> np.ndarray:
        """
        Predict expectation values for given parameters at time step t.
        
        Args:
            x: Circuit parameters of shape (n_samples, 2) for shared parameters
            t: Time step index
            
        Returns:
            Predicted expectation values of shape (n_qubit, n_samples)
        """
        if t == 0:
            return np.ones(shape=(self.n_qubit, x.shape[0]))
        
        if t % 2 == 0:
            return np.array([
                np.array(model[t].predict(
                    share_truncated_trigonometric_features_even(x, W[t], n_q=self.n_qubit)
                ))
                for model, W in zip(self.fitted_model, self.W_list)
            ])
        else:
            return np.array([
                np.array(model[t].predict(
                    share_truncated_trigonometric_features_odd(x, W[t], n_q=self.n_qubit)
                ))
                for model, W in zip(self.fitted_model, self.W_list)
            ])


class DTC_surrogate_share_truncation(DTC_surrogate_share):
    """DTC surrogate with explicit L0 truncation control for random features."""
    
    def build(
        self,
        n_trial_sample: int,
        num_feature: int,
        truncation: int,
        alpha: float,
        lr: float,
        epochs: int = 5000,
        note: Optional[str] = None,
        ep: bool = True,
        stop_threshold: float = 0.00001
    ) -> "DTC_surrogate_share_truncation":
        """
        Build surrogate model with explicit L0 truncation constraint.
        
        Args:
            n_trial_sample: Number of training samples
            num_feature: Number of random features
            truncation: Maximum number of non-zero elements per feature vector (L0 constraint)
            alpha: Ridge regularization strength
            lr: Learning rate
            epochs: Training epochs
            note: Optional note for saving files
            ep: Enable early stopping
            stop_threshold: Early stopping threshold
            
        Returns:
            Self for method chaining
        """
        self.n_trial_sample_ = n_trial_sample
        self.trial_params_ = self.params[-n_trial_sample:, :]
        self.trial_expvals_ = self.expvals[:, -n_trial_sample:, :]
        
        def get_random_W(nf, n_param, truncation_val, key=jax.random.key(1234)):
            truncation_val = min(truncation_val, n_param)
            if truncation_val == 0:
                return jnp.zeros((nf, n_param), dtype=jnp.int32)
            
            key_k, key_pos, key_sign = jax.random.split(key, 3)
            k_vals = jax.random.randint(key_k, (nf,), 0, truncation_val + 1)
            keys_pos = jax.random.split(key_pos, nf)
            
            def gen_pos(k, size, length_val):
                return jax.random.choice(k, length_val, shape=(size,), replace=False)
            
            indices = jax.vmap(gen_pos, in_axes=(0, None, None))(
                keys_pos, truncation_val, n_param
            )
            
            signs = jax.random.choice(
                key_sign, jnp.array([-1, 1]), shape=(nf, truncation_val)
            )
            
            vectors = jnp.zeros((nf, n_param), dtype=jnp.int32)
            
            def set_values(i, vecs):
                k = k_vals[i]
                if k > 0:
                    pos = indices[i, :k]
                    vals = signs[i, :k]
                    vecs = vecs.at[i, pos].set(vals)
                return vecs
            
            for i in range(nf):
                vectors = set_values(i, vectors)
            
            return vectors
        
        model_list = [[Ridgejax(alpha=alpha) for _ in range(self.n_T)] for _ in range(self.n_qubit)]
        thetas = np.zeros(shape=(self.n_qubit, self.n_T, num_feature))
        W_list = [[None for _ in range(self.n_T)] for _ in range(self.n_qubit)]
        
        for site in range(self.n_qubit):
            for T in range(self.n_T):
                print(f'Q{site+1}, T{T}')
                if T % 2 == 0:
                    n_unpack_params = (2 * self.n_qubit - 2) * (T // 2)
                    W = get_random_W(num_feature, n_unpack_params, truncation)
                    model_list[site][T].fit(
                        features=share_truncated_trigonometric_features_even(
                            self.trial_params_, W, n_q=self.n_qubit
                        ),
                        y=self.trial_expvals_[site, :, T],
                        lr=lr,
                        epochs=epochs,
                        ep=ep,
                        stop_threshold=stop_threshold,
                        display=False
                    )
                else:
                    n_unpack_params = (2 * self.n_qubit - 2) * ((T - 1) // 2) + self.n_qubit
                    W = get_random_W(num_feature, n_unpack_params, truncation)
                    model_list[site][T].fit(
                        features=share_truncated_trigonometric_features_odd(
                            self.trial_params_, W, n_q=self.n_qubit
                        ),
                        y=self.trial_expvals_[site, :, T],
                        lr=lr,
                        epochs=epochs,
                        ep=ep,
                        stop_threshold=stop_threshold,
                        display=False
                    )
                
                thetas[site][T] = np.array(model_list[site][T].theta_)
                W_list[site][T] = W
        
        if note is not None:
            np.save(
                f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/thetas_{self.n_qubit}q_'
                f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}p_{note}.npy',
                thetas
            )
            with open(
                f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/Ws_{self.n_qubit}q_'
                f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}p_{note}.npy',
                'wb'
            ) as f:
                pickle.dump(W_list, f)
        
        self.fitted_model = model_list
        self.W_list = W_list
        return self
    
    # Inherit finetune, load, and predict methods from parent class


class Ridgejax_fix_epoches(Ridgejax):
    """Ridge regression with fixed number of training epochs (no early stopping)."""
    
    def __init__(self, alpha: float = 10.0, theta: Optional[jnp.ndarray] = None):
        super().__init__(alpha, theta)
        self.x_his = []
        self.loss_his = []
    
    def fit(
        self,
        features: jnp.ndarray,
        y: jnp.ndarray,
        lr: float = 0.01,
        epochs: int = 3000
    ) -> "Ridgejax_fix_epoches":
        """
        Fit model for exactly the specified number of epochs.
        
        Args:
            features: Training features
            y: Target values
            lr: Learning rate
            epochs: Exact number of training epochs
            
        Returns:
            Self for method chaining
        """
        @jax.jit
        def loss(theta, features_arr, y_arr, alpha_val):
            pred = jnp.matmul(theta, features_arr.T)
            return jnp.sum(jnp.square(pred - y_arr)) + alpha_val * jnp.linalg.norm(theta, ord=2)
        
        theta = jax.random.uniform(
            jax.random.key(1234), (features.shape[1],), minval=-0.1, maxval=0.1
        )
        opt = optax.adam(learning_rate=lr)
        opt_state = opt.init(theta)
        value_and_grad_fn = jax.jit(jax.value_and_grad(loss))
        alpha_val = self.alpha
        
        for i in range(epochs):
            loss_val, grads = value_and_grad_fn(theta, features, y, alpha_val)
            self.x_his.append(theta)
            self.loss_his.append(loss_val)
            updates, opt_state = opt.update(grads, opt_state, theta)
            theta = optax.apply_updates(theta, updates)
        
        self.theta_ = self.x_his[-1]
        print(f'Epochs:{epochs}, Loss:{loss_val:.6f}')
        return self


class DTC_surrogate_share_evahmb(DTC_surrogate_share):
    """DTC surrogate with history tracking for evaluation and hyperparameter benchmarking."""
    
    def build(
        self,
        n_trial_sample: int,
        num_feature: int,
        truncation: int,
        alpha: float,
        note: str,
        epoch: int = 300,
        rng: int = 1234
    ) -> "DTC_surrogate_share_evahmb":
        """
        Build surrogate model with training history tracking.
        
        Args:
            n_trial_sample: Number of training samples
            num_feature: Number of random features
            truncation: L0 truncation constraint
            alpha: Ridge regularization strength
            note: Note for saving files
            epoch: Fixed training epochs per model
            rng: Random seed
            
        Returns:
            Self for method chaining
        """
        self.n_trial_sample_ = n_trial_sample
        self.trial_params_ = self.params[-n_trial_sample:, :]
        self.trial_expvals_ = self.expvals[:, -n_trial_sample:, :]
        
        model_list = [
            [Ridgejax_fix_epoches(alpha=alpha) for _ in range(self.n_T)]
            for _ in range(self.n_qubit)
        ]
        transposed_model_list = list(map(list, zip(*model_list)))
        thetas = np.zeros(shape=(self.n_qubit, self.n_T, num_feature))
        W_list = [[None for _ in range(self.n_T)] for _ in range(self.n_qubit)]
        loss_his_list = [[None for _ in range(self.n_T)] for _ in range(self.n_qubit)]
        thetas_his_list = [[None for _ in range(self.n_T)] for _ in range(self.n_qubit)]
        
        for T, model_overT in enumerate(transposed_model_list):
            if T % 2 == 0:
                n_unpack_params = (2 * self.n_qubit - 2) * (T // 2)
                W = get_w_vectors_random(
                    length=n_unpack_params,
                    max_nonzero=truncation,
                    nf=num_feature,
                    key=jax.random.key(rng)
                )
                for site, model in enumerate(model_overT):
                    model.fit(
                        features=share_truncated_trigonometric_features_even(
                            self.trial_params_, W, n_q=self.n_qubit
                        ),
                        y=self.trial_expvals_[site, :, T],
                        lr=0.02,
                        epochs=epoch
                    )
                    print(f'T:{T+1}, Q index{site+1}')
                    thetas[site][T] = np.array(model.theta_)
                    W_list[site][T] = W
                    loss_his_list[site][T] = np.array(model.loss_his)
                    thetas_his_list[site][T] = np.array(model.x_his)
            else:
                n_unpack_params = (2 * self.n_qubit - 2) * ((T - 1) // 2) + self.n_qubit
                W = get_w_vectors_random(
                    length=n_unpack_params,
                    max_nonzero=truncation,
                    nf=num_feature,
                    key=jax.random.key(rng)
                )
                for site, model in enumerate(model_overT):
                    model.fit(
                        features=share_truncated_trigonometric_features_odd(
                            self.trial_params_, W, n_q=self.n_qubit
                        ),
                        y=self.trial_expvals_[site, :, T],
                        lr=0.02,
                        epochs=epoch
                    )
                    print(f'T:{T+1}, Q index{site+1}')
                    thetas[site][T] = np.array(model.theta_)
                    W_list[site][T] = W
                    loss_his_list[site][T] = np.array(model.loss_his)
                    thetas_his_list[site][T] = np.array(model.x_his)
        
        np.save(
            f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/loss_his_{self.n_qubit}q_'
            f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}lambda_{note}_seed({rng}).npy',
            np.array(loss_his_list)
        )
        np.save(
            f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/thetas_his_{self.n_qubit}q_'
            f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}lambda_{note}_seed({rng}).npy',
            np.array(thetas_his_list)
        )
        np.save(
            f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/thetas_{self.n_qubit}q_'
            f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}lambda_{note}_seed({rng}).npy',
            thetas
        )
        with open(
            f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/Ws_{self.n_qubit}q_'
            f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}lambda_{note}_seed({rng}).npy',
            'wb'
        ) as f:
            pickle.dump(W_list, f)
        
        self.fitted_model = list(map(list, zip(*transposed_model_list)))
        self.W_list = W_list
        return self
    
    # Inherit load and predict methods from parent class


class DTC_surrogate_share_evahmb_rp(DTC_surrogate_share_evahmb):
    """DTC surrogate with random probability-based feature generation (alternative to L0 truncation)."""
    
    def build(
        self,
        n_trial_sample: int,
        num_feature: int,
        truncation: float,
        alpha: float,
        note: str,
        epoch: int = 300,
        rng: int = 1234
    ) -> "DTC_surrogate_share_evahmb_rp":
        """
        Build surrogate model using random probability for feature generation.
        
        Args:
            n_trial_sample: Number of training samples
            num_feature: Number of random features
            truncation: Probability parameter for non-zero elements (p1)
            alpha: Ridge regularization strength
            note: Note for saving files
            epoch: Fixed training epochs per model
            rng: Random seed
            
        Returns:
            Self for method chaining
        """
        def get_random_W(num_feature_val, num_param, p1, key):
            probabilities = jnp.array([p1, 1 - 2 * p1, p1])
            return jax.random.choice(
                key,
                jnp.array([-1, 0, 1]),
                shape=(num_feature_val, num_param),
                replace=True,
                p=probabilities
            )
        
        self.n_trial_sample_ = n_trial_sample
        self.trial_params_ = self.params[-n_trial_sample:, :]
        self.trial_expvals_ = self.expvals[:, -n_trial_sample:, :]
        
        model_list = [
            [Ridgejax_fix_epoches(alpha=alpha) for _ in range(self.n_T)]
            for _ in range(self.n_qubit)
        ]
        transposed_model_list = list(map(list, zip(*model_list)))
        thetas = np.zeros(shape=(self.n_qubit, self.n_T, num_feature))
        W_list = [[None for _ in range(self.n_T)] for _ in range(self.n_qubit)]
        loss_his_list = [[None for _ in range(self.n_T)] for _ in range(self.n_qubit)]
        thetas_his_list = [[None for _ in range(self.n_T)] for _ in range(self.n_qubit)]
        
        for T, model_overT in enumerate(transposed_model_list):
            if T % 2 == 0:
                n_unpack_params = (2 * self.n_qubit - 2) * (T // 2)
                W = get_random_W(
                    num_feature=num_feature,
                    num_param=n_unpack_params,
                    p1=truncation,
                    key=jax.random.key(rng)
                )
                for site, model in enumerate(model_overT):
                    model.fit(
                        features=share_truncated_trigonometric_features_even(
                            self.trial_params_, W, n_q=self.n_qubit
                        ),
                        y=self.trial_expvals_[site, :, T],
                        lr=0.02,
                        epochs=epoch
                    )
                    print(f'T:{T+1}, Q index{site+1}')
                    thetas[site][T] = np.array(model.theta_)
                    W_list[site][T] = W
                    loss_his_list[site][T] = np.array(model.loss_his)
                    thetas_his_list[site][T] = np.array(model.x_his)
            else:
                n_unpack_params = (2 * self.n_qubit - 2) * ((T - 1) // 2) + self.n_qubit
                W = get_random_W(
                    num_feature=num_feature,
                    num_param=n_unpack_params,
                    p1=truncation,
                    key=jax.random.key(rng)
                )
                for site, model in enumerate(model_overT):
                    model.fit(
                        features=share_truncated_trigonometric_features_odd(
                            self.trial_params_, W, n_q=self.n_qubit
                        ),
                        y=self.trial_expvals_[site, :, T],
                        lr=0.02,
                        epochs=epoch
                    )
                    print(f'T:{T+1}, Q index{site+1}')
                    thetas[site][T] = np.array(model.theta_)
                    W_list[site][T] = W
                    loss_his_list[site][T] = np.array(model.loss_his)
                    thetas_his_list[site][T] = np.array(model.x_his)
        
        np.save(
            f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/loss_his_{self.n_qubit}q_'
            f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}lambda_{note}_seed({rng}).npy',
            np.array(loss_his_list)
        )
        np.save(
            f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/thetas_his_{self.n_qubit}q_'
            f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}lambda_{note}_seed({rng}).npy',
            np.array(thetas_his_list)
        )
        np.save(
            f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/thetas_{self.n_qubit}q_'
            f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}lambda_{note}_seed({rng}).npy',
            thetas
        )
        with open(
            f'/home/lwy/shadow/Fig4/expdata/Ws_thetas/Ws_{self.n_qubit}q_'
            f'{self.n_trial_sample_}samples_{num_feature}feature_{alpha}a_{truncation}lambda_{note}_seed({rng}).npy',
            'wb'
        ) as f:
            pickle.dump(W_list, f)
        
        self.fitted_model = list(map(list, zip(*transposed_model_list)))
        self.W_list = W_list
        return self
    
    # Inherit load and predict methods from parent class