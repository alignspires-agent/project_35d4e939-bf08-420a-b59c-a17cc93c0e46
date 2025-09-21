
import numpy as np
import sys
import logging
from typing import Tuple, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LevyProkhorovConformalPrediction:
    """
    Implementation of Conformal Prediction under Lévy-Prokhorov Distribution Shifts
    based on the paper methodology.
    """
    
    def __init__(self, epsilon: float, rho: float, alpha: float = 0.1):
        """
        Initialize the LP conformal prediction model.
        
        Parameters:
        epsilon (float): Local perturbation parameter (ε)
        rho (float): Global perturbation parameter (ρ)
        alpha (float): Significance level (1 - coverage level)
        """
        self.epsilon = epsilon
        self.rho = rho
        self.alpha = alpha
        self.scores = None
        self.quantile = None
        
    def compute_scores(self, X_calib: np.ndarray, y_calib: np.ndarray) -> np.ndarray:
        """
        Compute conformity scores for calibration data.
        Using absolute error as a simple scoring function.
        
        Parameters:
        X_calib (np.ndarray): Calibration features
        y_calib (np.ndarray): Calibration targets
        
        Returns:
        np.ndarray: Conformity scores
        """
        try:
            # Simple scoring function: absolute error
            # In practice, this would be replaced with model-specific scoring
            scores = np.abs(y_calib - np.mean(X_calib, axis=1))
            return scores
        except Exception as e:
            logger.error(f"Error computing scores: {e}")
            sys.exit(1)
    
    def compute_worst_case_quantile(self, scores: np.ndarray) -> float:
        """
        Compute the worst-case quantile under LP distribution shifts.
        
        Parameters:
        scores (np.ndarray): Conformity scores
        
        Returns:
        float: Worst-case quantile value
        """
        try:
            n = len(scores)
            sorted_scores = np.sort(scores)
            
            # Compute empirical quantile
            empirical_quantile_idx = int(np.ceil((1 - self.alpha) * (n + 1))) - 1
            empirical_quantile = sorted_scores[empirical_quantile_idx]
            
            # Apply LP robustness adjustments
            # Based on the paper's theoretical results for worst-case quantile
            worst_case_quantile = empirical_quantile + self.epsilon + self.rho
            
            return worst_case_quantile
        except Exception as e:
            logger.error(f"Error computing worst-case quantile: {e}")
            sys.exit(1)
    
    def fit(self, X_calib: np.ndarray, y_calib: np.ndarray):
        """
        Fit the conformal prediction model on calibration data.
        
        Parameters:
        X_calib (np.ndarray): Calibration features
        y_calib (np.ndarray): Calibration targets
        """
        try:
            logger.info("Fitting LP conformal prediction model...")
            self.scores = self.compute_scores(X_calib, y_calib)
            self.quantile = self.compute_worst_case_quantile(self.scores)
            logger.info(f"Model fitted successfully. Worst-case quantile: {self.quantile:.4f}")
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            sys.exit(1)
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals for test data.
        
        Parameters:
        X_test (np.ndarray): Test features
        
        Returns:
        Tuple[np.ndarray, np.ndarray]: Lower and upper bounds of prediction intervals
        """
        try:
            if self.quantile is None:
                raise ValueError("Model must be fitted before prediction")
            
            # Simple prediction: mean of features as point estimate
            point_estimates = np.mean(X_test, axis=1)
            
            # Create prediction intervals
            lower_bounds = point_estimates - self.quantile
            upper_bounds = point_estimates + self.quantile
            
            return lower_bounds, upper_bounds
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            sys.exit(1)
    
    def evaluate_coverage(self, y_true: np.ndarray, lower_bounds: np.ndarray, 
                         upper_bounds: np.ndarray) -> float:
        """
        Evaluate coverage of prediction intervals.
        
        Parameters:
        y_true (np.ndarray): True target values
        lower_bounds (np.ndarray): Lower bounds of intervals
        upper_bounds (np.ndarray): Upper bounds of intervals
        
        Returns:
        float: Coverage percentage
        """
        try:
            coverage = np.mean((y_true >= lower_bounds) & (y_true <= upper_bounds))
            return coverage
        except Exception as e:
            logger.error(f"Error evaluating coverage: {e}")
            sys.exit(1)

def generate_synthetic_data(n_samples: int = 1000, n_features: int = 10, 
                          noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for experimentation.
    
    Parameters:
    n_samples (int): Number of samples
    n_features (int): Number of features
    noise_level (float): Noise level in target
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: Features and targets
    """
    try:
        np.random.seed(42)  # For reproducibility
        
        # Generate features from normal distribution
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Generate targets as linear combination of features plus noise
        true_coef = np.random.normal(0, 1, n_features)
        y = X @ true_coef + np.random.normal(0, noise_level, n_samples)
        
        return X, y
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        sys.exit(1)

def simulate_distribution_shift(X: np.ndarray, y: np.ndarray, 
                              shift_strength: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate distribution shift for testing robustness.
    
    Parameters:
    X (np.ndarray): Original features
    y (np.ndarray): Original targets
    shift_strength (float): Strength of distribution shift
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: Shifted features and targets
    """
    try:
        X_shifted = X + np.random.normal(0, shift_strength, X.shape)
        y_shifted = y + np.random.normal(0, shift_strength, y.shape)
        
        return X_shifted, y_shifted
    except Exception as e:
        logger.error(f"Error simulating distribution shift: {e}")
        sys.exit(1)

def main():
    """
    Main experiment function to evaluate the LP conformal prediction method.
    """
    logger.info("Starting LP Conformal Prediction Experiment")
    
    try:
        # Generate synthetic data
        logger.info("Generating synthetic data...")
        X, y = generate_synthetic_data(n_samples=1000, n_features=5)
        
        # Split data into calibration and test sets
        split_idx = int(0.7 * len(X))
        X_calib, X_test = X[:split_idx], X[split_idx:]
        y_calib, y_test = y[:split_idx], y[split_idx:]
        
        # Initialize and fit LP conformal prediction model
        lp_cp = LevyProkhorovConformalPrediction(epsilon=0.1, rho=0.05, alpha=0.1)
        lp_cp.fit(X_calib, y_calib)
        
        # Generate predictions on test data
        lower_bounds, upper_bounds = lp_cp.predict(X_test)
        
        # Evaluate coverage on test data
        coverage = lp_cp.evaluate_coverage(y_test, lower_bounds, upper_bounds)
        logger.info(f"Coverage on test data: {coverage:.4f}")
        
        # Test robustness under distribution shift
        logger.info("Testing robustness under distribution shift...")
        X_test_shifted, y_test_shifted = simulate_distribution_shift(X_test, y_test, shift_strength=0.3)
        
        # Generate predictions on shifted test data
        lower_bounds_shifted, upper_bounds_shifted = lp_cp.predict(X_test_shifted)
        
        # Evaluate coverage on shifted test data
        coverage_shifted = lp_cp.evaluate_coverage(y_test_shifted, lower_bounds_shifted, upper_bounds_shifted)
        logger.info(f"Coverage on shifted test data: {coverage_shifted:.4f}")
        
        # Compare with non-robust conformal prediction (epsilon=0, rho=0)
        logger.info("Comparing with non-robust conformal prediction...")
        non_robust_cp = LevyProkhorovConformalPrediction(epsilon=0.0, rho=0.0, alpha=0.1)
        non_robust_cp.fit(X_calib, y_calib)
        
        lower_bounds_nr, upper_bounds_nr = non_robust_cp.predict(X_test_shifted)
        coverage_nr_shifted = non_robust_cp.evaluate_coverage(y_test_shifted, lower_bounds_nr, upper_bounds_nr)
        logger.info(f"Non-robust coverage on shifted data: {coverage_nr_shifted:.4f}")
        
        # Print final results
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*60)
        print(f"LP Robust Conformal Prediction:")
        print(f"  - Epsilon (local perturbation): {lp_cp.epsilon}")
        print(f"  - Rho (global perturbation): {lp_cp.rho}")
        print(f"  - Target coverage: {1 - lp_cp.alpha:.1%}")
        print(f"  - Achieved coverage (original): {coverage:.4f} ({coverage:.1%})")
        print(f"  - Achieved coverage (shifted): {coverage_shifted:.4f} ({coverage_shifted:.1%})")
        print(f"  - Worst-case quantile: {lp_cp.quantile:.4f}")
        print()
        print(f"Non-Robust Conformal Prediction:")
        print(f"  - Achieved coverage (shifted): {coverage_nr_shifted:.4f} ({coverage_nr_shifted:.1%})")
        print()
        print(f"Robustness Improvement: {abs(coverage_shifted - (1 - lp_cp.alpha)):.4f} "
              f"vs {abs(coverage_nr_shifted - (1 - lp_cp.alpha)):.4f}")
        print("="*60)
        
        # Return success
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
