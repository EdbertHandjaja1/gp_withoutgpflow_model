import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class GaussianProcessModel:
    def __init__(self, kernel, noise_variance=1.0):        
        """
        A Gaussian Process Model: Y_i = f(x_i) + ε_i
            - f(x_i) is a function with a Gaussian Process prior
            - The additive noise (ε_i) is i.i.d. Gaussian noise with zero mean and variance noise_variance

        Args:
            kernel: the GP model's covariance function (kernel object, not int)
            noise_variance (float): the variance of the observation noise (must be non-negative)
        """
        self.kernel = kernel
        self.noise_variance = tf.Variable(noise_variance, name="noise_variance", dtype=tf.float64)
        self.X_train = None
        self.y_train = None
        self.L = None

    def _negative_log_marginal_likelihood(self, variance, length_scale, noise_variance):
        """
        Calculate the negative log marginal likelihood.

        Args:
            variance (float): Kernel variance parameter
            length_scale (float): Kernel length scale parameter
            noise_variance (float): Noise variance parameter (σ²)

        Returns:
            float: Negative log marginal likelihood value
        """
        self.kernel.variance.assign(variance)
        self.kernel.length_scale.assign(length_scale)
        self.noise_variance.assign(noise_variance)

        # K + σ²I
        K_plus_noise = self.kernel(self.X_train, self.X_train) + \
            self.noise_variance * tf.eye(tf.shape(self.X_train)[0], dtype=tf.float64)
        

        # L = cholesky(K + σ²I)
        L = tf.linalg.cholesky(K_plus_noise)

        # α = L^T \ (L\y)
        alpha = tf.linalg.triangular_solve(L, self.y_train, lower=True)
        alpha = tf.linalg.triangular_solve(tf.transpose(L), alpha, lower=False)

        first_term = 0.5 * tf.linalg.matmul(tf.transpose(self.y_train), alpha)
        second_term = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        third_term = 0.5 * tf.cast(tf.shape(self.X_train)[0], tf.double) * tf.math.log(tf.constant(2.0, dtype=tf.double) * np.pi)

        negative_log_likelihood = (first_term + second_term + third_term)
        
        return negative_log_likelihood.numpy()

    def fit(self, X_train, y_train):
        """
        Fits the Gaussian Process model to the training data by optimizing the kernel parameters.

        Args:
            X (np.ndarray): Input of shape (N, 1), where N is the number of samples, 1 because its for regression
            y (np.ndarray): Target values of shape (N, 1) corresponding to X

        Returns:
            int: Number of optimization iterations (converged iterations or max_iter).
        """
        self.X_train = tf.convert_to_tensor(X_train, dtype=tf.float64)
        self.y_train = tf.convert_to_tensor(y_train, dtype=tf.float64)

        # Define the function for minimization
        def objective_function(params):
            variance, length_scale, noise_variance = params
            return self._negative_log_marginal_likelihood(variance, length_scale, noise_variance)
        
        initial_params = [
            self.kernel.variance.numpy(),  
            self.kernel.length_scale.numpy(),
            self.noise_variance.numpy()
        ]

        results = minimize(
            fun=objective_function,
            x0=initial_params,   
            method='L-BFGS-B',
            bounds=[(1e-5, None), (1e-5, None), (1e-5, None)] 
        )

        self.kernel.variance.assign(results.x[0])
        self.kernel.length_scale.assign(results.x[1])
        self.noise_variance.assign(results.x[2])
        
        # K + σ²I
        K_plus_noise = self.kernel(self.X_train, self.X_train) + \
            self.noise_variance * tf.eye(tf.shape(self.X_train)[0], dtype=tf.float64)

        # L = cholesky(K + σ²I)
        self.L = tf.linalg.cholesky(K_plus_noise)

        print(f"\nOptimized parameters: Variance={self.kernel.variance.numpy():.4f}, "
              f"Length Scale={self.kernel.length_scale.numpy():.4f}, "
              f"Noise Variance={self.noise_variance.numpy():.4f}")
        
        # Optimized parameters: Variance=0.5703, Length Scale=1.9191, Noise Variance=0.2024
        # self.kernel.variance.assign(0.5703)
        # self.kernel.length_scale.assign(1.9191)
        # self.noise_variance.assign(0.2024)

    def predict(self, X_test):
        """
        Calculate predicted mean and variance

        Args:
            X_test (np.ndarray): Input of shape (N, 1), where N is the number of samples, 1 because its for regression

        Returns:
            int: Number of optimization iterations (converged iterations or max_iter).
        """
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float64)

        k_star = self.kernel(self.X_train, X_test)
        k_star_star = self.kernel(X_test, X_test)

        # α = L^T \ (L\y)
        alpha = tf.linalg.triangular_solve(self.L, self.y_train, lower=True)
        alpha = tf.linalg.triangular_solve(tf.transpose(self.L), alpha, lower=False)

        # predictive mean = k_star^T * α
        pred_mean = tf.linalg.matmul(tf.transpose(k_star), alpha)

        # v = L \ k_star
        v = tf.linalg.triangular_solve(self.L, k_star, lower=True)

        # predictive variance = k_star_star - v^T * v
        pred_var = k_star_star - tf.linalg.matmul(tf.transpose(v), v)

        return pred_mean, tf.linalg.diag_part(pred_var).numpy()
        

# initialize kernel 
class Matern32Kernel:
    def __init__(self, variance, length_scale):
        """
        Matern 3/2 Kernel.
        Args:
            variance (float): Signal variance (σ²).
            length_scale (float): Length scale (ℓ).
        """
        self.variance = tf.Variable(variance, name="variance", dtype=tf.float64)
        self.length_scale = tf.Variable(length_scale, name="length_scale", dtype=tf.float64)

    def __call__(self, X1, X2):
        """
        Computes the covariance matrix between X1 and X2, with r scaled by ℓ
        Args:
            X1: Array of shape (n_samples_1, n_features).
            X2: Array of shape (n_samples_2, n_features).
        Returns:
            Covariance matrix of shape (n_samples_1, n_samples_2).
        """
        sq_dist = tf.reduce_sum(tf.square(X1[:, None] - X2), axis=-1)
        r = tf.sqrt(sq_dist) / self.length_scale
        return self._matern32(r)

    def _matern32(self, r):
        """
        Matern 3/2 kernel function.
        Source: TensorFlow Framework
        Args:
            r: the Euclidean distance between the input points
        Returns:
            Matern 3/2 Kernel with euclidian distance r
        """
        sqrt3 = np.sqrt(3.0)
        return self.variance * (1.0 + sqrt3 * r) * tf.exp(-sqrt3 * r)

# Example usage
kernel = Matern32Kernel(variance=1.0, length_scale=1.0)
gp = GaussianProcessModel(kernel)

X_train = np.linspace(0, 10, 100).reshape(-1, 1)
y_train = np.sin(X_train) + 0.5 * np.random.randn(*X_train.shape)

gp.fit(X_train, y_train)

X_test = np.linspace(-2, 12, 200).reshape(-1, 1)
mean, var = gp.predict(X_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'kx', alpha=0.6, label='Observations')
plt.plot(X_test, mean, 'b-', label='Predicted mean')

mean_np = mean.numpy().flatten()  
var_np = var

plt.fill_between(X_test.flatten(),
                 (mean_np - 2 * np.sqrt(var_np)),
                 (mean_np + 2 * np.sqrt(var_np)),
                 alpha=0.2, color='blue', label='95% Confidence Interval')

plt.xlabel('Input X')
plt.ylabel('Output Y')
plt.title('Gaussian Process Regression with Matern 3/2 Kernel')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()