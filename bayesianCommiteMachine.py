import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic, DotProduct

class BayesianCommitteeMachine:
    def __init__(self, n_subsets=5, kernel=None, gpr_n_restarts_optimizer=10, gpr_alpha=1e-10, gpr_normalize_y=False, gpr_copy_X_train=True, gpr_random_state=None):
        """
        Bayesian Committee Machine (BCM) constructor.

        :param n_subsets: Number of subsets to split the data into.
        :param kernel: Kernel to be used in Gaussian Process Regressor. Can be user-defined or default.
        :param gpr_n_restarts_optimizer: Number of restarts for optimizer.
        :param gpr_alpha: Value added to the diagonal of the kernel matrix during fitting.
        :param gpr_normalize_y: Whether to normalize the target values.
        :param gpr_copy_X_train: Whether to copy X_train during fitting.
        :param gpr_random_state: Seed used by the random number generator.
        """
        self.n_subsets = n_subsets
        
        # Default kernel is RBF kernel if not provided by the user
        self.kernel = kernel if kernel else C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        
        # Store GaussianProcessRegressor parameters with gpr_ prefix
        self.gpr_n_restarts_optimizer = gpr_n_restarts_optimizer
        self.gpr_alpha = gpr_alpha
        self.gpr_normalize_y = gpr_normalize_y
        self.gpr_copy_X_train = gpr_copy_X_train
        self.gpr_random_state = gpr_random_state
        
        self.models = []
        self.subset_data = []

    def split_data(self, X, y):
        """
        Split data into smaller subsets.

        :param X: Input data.
        :param y: Target values.
        """
        data_size = len(X)
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        subset_size = data_size // self.n_subsets

        for i in range(self.n_subsets):
            subset_indices = indices[i*subset_size:(i+1)*subset_size]
            self.subset_data.append((X[subset_indices], y[subset_indices]))

    def train_models(self):
        """
        Train a Gaussian Process Regressor for each subset.
        """
        for X_sub, y_sub in self.subset_data:
            model = GaussianProcessRegressor(
                kernel=self.kernel,
                n_restarts_optimizer=self.gpr_n_restarts_optimizer,
                alpha=self.gpr_alpha,
                normalize_y=self.gpr_normalize_y,
                copy_X_train=self.gpr_copy_X_train,
                random_state=self.gpr_random_state
            )
            model.fit(X_sub, y_sub)
            self.models.append(model)

    def predict(self, X):
        """
        Combine predictions from all models with uncertainty.

        :param X: Input data to predict.
        :return: Weighted combined predictions from all models.
        """
        predictions = []
        variances = []

        for model in self.models:
            pred, sigma = model.predict(X, return_std=True)
            predictions.append(pred)
            variances.append(sigma**2)

        # Convert to numpy arrays for easier manipulation
        predictions = np.array(predictions)
        variances = np.array(variances)

        # Compute weights as the inverse of variance (uncertainty)
        weights = 1 / variances

        # Weighted average of predictions
        weighted_predictions = np.sum(predictions * weights, axis=0) / np.sum(weights, axis=0)

        # Combine uncertainties
        combined_variances = 1 / np.sum(weights, axis=0)

        return weighted_predictions, np.sqrt(combined_variances)

    def fit(self, X, y):
        """
        Fit the BCM model by splitting data and training models.

        :param X: Input data.
        :param y: Target values.
        """
        self.split_data(X, y)
        self.train_models()

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel()

    # Kernel options
    rbf_kernel = C(1.0) * RBF(length_scale=1.0)
    matern_kernel = Matern(nu=1.5)
    rq_kernel = RationalQuadratic(alpha=0.1, length_scale=1.0)
    dot_product_kernel = DotProduct()

    # Initialize BCM with different kernels and GaussianProcessRegressor parameters
    bcm_rbf = BayesianCommitteeMachine(n_subsets=3, kernel=rbf_kernel, gpr_n_restarts_optimizer=5, gpr_alpha=1e-5)
    bcm_matern = BayesianCommitteeMachine(n_subsets=3, kernel=matern_kernel, gpr_normalize_y=True)
    bcm_rq = BayesianCommitteeMachine(n_subsets=3, kernel=rq_kernel, gpr_n_restarts_optimizer=15)
    bcm_dot = BayesianCommitteeMachine(n_subsets=3, kernel=dot_product_kernel)

    # Fit the models
    bcm_rbf.fit(X, y)
    bcm_matern.fit(X, y)
    bcm_rq.fit(X, y)
    bcm_dot.fit(X, y)

    # Predict
    X_test = np.linspace(0, 10, 50).reshape(-1, 1)
    y_pred_rbf, y_uncertainty_rbf = bcm_rbf.predict(X_test)
    y_pred_matern, y_uncertainty_matern = bcm_matern.predict(X_test)
    y_pred_rq, y_uncertainty_rq = bcm_rq.predict(X_test)
    y_pred_dot, y_uncertainty_dot = bcm_dot.predict(X_test)

    # Display results
    print("RBF Kernel Predictions:", y_pred_rbf)
    print("RBF Kernel Uncertainty:", y_uncertainty_rbf)

    print("Matern Kernel Predictions:", y_pred_matern)
    print("Matern Kernel Uncertainty:", y_uncertainty_matern)

    print("Rational Quadratic Kernel Predictions:", y_pred_rq)
    print("Rational Quadratic Kernel Uncertainty:", y_uncertainty_rq)

    print("Dot Product Kernel Predictions:", y_pred_dot)
    print("Dot Product Kernel Uncertainty:", y_uncertainty_dot)
