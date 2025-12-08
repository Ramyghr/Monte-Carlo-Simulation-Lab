"""
Enhanced Monte Carlo Utilities
Advanced variance reduction, convergence analysis, and performance tracking
"""

import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky
import time
from typing import Dict, Tuple, List, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


class VarianceReduction:
    """Advanced variance reduction techniques"""
    
    @staticmethod
    def antithetic_variates(func: Callable, n_samples: int, *args, **kwargs) -> Dict:
        """
        Antithetic variates variance reduction
        Returns both original and antithetic samples
        """
        start_time = time.time()
        
        # Generate half the samples
        samples = []
        antithetic_samples = []
        
        for _ in range(n_samples // 2):
            z = np.random.randn()
            sample = func(z, *args, **kwargs)
            anti_sample = func(-z, *args, **kwargs)
            
            samples.append(sample)
            antithetic_samples.append(anti_sample)
        
        all_samples = samples + antithetic_samples
        
        return {
            'mean': np.mean(all_samples),
            'std': np.std(all_samples),
            'variance': np.var(all_samples),
            'samples': all_samples,
            'time': time.time() - start_time,
            'variance_reduction': 1 - (np.var(all_samples) / np.var(samples)) if np.var(samples) > 0 else 0
        }
    
    @staticmethod
    def control_variates(payoffs: np.ndarray, control: np.ndarray, 
                        control_mean: float) -> Dict:
        """
        Control variates variance reduction
        Uses a correlated variable with known expectation
        """
        # Calculate optimal coefficient
        cov = np.cov(payoffs, control)[0, 1]
        var_control = np.var(control)
        
        if var_control > 0:
            theta = cov / var_control
        else:
            theta = 0
        
        # Adjusted payoffs
        adjusted = payoffs - theta * (control - control_mean)
        
        return {
            'mean': np.mean(adjusted),
            'std': np.std(adjusted),
            'variance': np.var(adjusted),
            'theta': theta,
            'original_variance': np.var(payoffs),
            'variance_reduction': 1 - (np.var(adjusted) / np.var(payoffs)) if np.var(payoffs) > 0 else 0
        }
    
    @staticmethod
    def importance_sampling(func: Callable, n_samples: int, 
                          shift: float, *args, **kwargs) -> Dict:
        """
        Importance sampling for rare event simulation
        Shifts the distribution to sample more relevant regions
        """
        start_time = time.time()
        
        samples = []
        weights = []
        
        for _ in range(n_samples):
            # Sample from shifted distribution
            z = np.random.randn() + shift
            
            # Calculate likelihood ratio (Radon-Nikodym derivative)
            weight = np.exp(-shift * z + 0.5 * shift**2)
            
            sample = func(z, *args, **kwargs)
            samples.append(sample * weight)
            weights.append(weight)
        
        weighted_mean = np.sum(samples) / n_samples
        
        return {
            'mean': weighted_mean,
            'samples': samples,
            'weights': weights,
            'time': time.time() - start_time
        }


class ConvergenceAnalyzer:
    """Convergence analysis and error estimation"""
    
    @staticmethod
    def analyze_convergence(func: Callable, true_value: float, 
                          sample_sizes: List[int], n_trials: int = 20) -> Dict:
        """
        Analyze convergence behavior across different sample sizes
        """
        results = []
        
        for n in sample_sizes:
            trial_estimates = []
            trial_times = []
            
            for _ in range(n_trials):
                start = time.time()
                estimate = func(n)
                trial_times.append(time.time() - start)
                trial_estimates.append(estimate)
            
            estimates = np.array(trial_estimates)
            
            results.append({
                'n': n,
                'mean_estimate': np.mean(estimates),
                'std_estimate': np.std(estimates),
                'mean_error': np.mean(np.abs(estimates - true_value)),
                'rmse': np.sqrt(np.mean((estimates - true_value)**2)),
                'bias': np.mean(estimates - true_value),
                'theoretical_std': 1.0 / np.sqrt(n),  # O(1/√n) convergence
                'mean_time': np.mean(trial_times),
                'efficiency': 1.0 / (np.var(estimates) * np.mean(trial_times))
            })
        
        return {
            'results': results,
            'true_value': true_value,
            'n_trials': n_trials
        }
    
    @staticmethod
    def confidence_interval(samples: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for Monte Carlo estimate"""
        mean = np.mean(samples)
        std_error = np.std(samples) / np.sqrt(len(samples))
        z_score = norm.ppf((1 + confidence) / 2)
        
        ci_lower = mean - z_score * std_error
        ci_upper = mean + z_score * std_error
        
        return ci_lower, ci_upper


class StochasticProcesses:
    """Enhanced stochastic process simulations"""
    
    @staticmethod
    def gbm_paths(S0: float, mu: float, sigma: float, T: float, 
                  steps: int, paths: int, scheme: str = 'exact') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate GBM paths using various schemes
        schemes: 'exact', 'euler', 'milstein'
        """
        dt = T / steps
        t = np.linspace(0, T, steps + 1)
        S = np.zeros((paths, steps + 1))
        S[:, 0] = S0
        
        if scheme == 'exact':
            # Exact solution (fastest and most accurate)
            for i in range(steps):
                Z = np.random.randn(paths)
                S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        elif scheme == 'euler':
            for i in range(steps):
                Z = np.random.randn(paths)
                dW = np.sqrt(dt) * Z
                S[:, i + 1] = S[:, i] * (1 + mu * dt + sigma * dW)
        
        elif scheme == 'milstein':
            for i in range(steps):
                Z = np.random.randn(paths)
                dW = np.sqrt(dt) * Z
                S[:, i + 1] = S[:, i] * (1 + mu * dt + sigma * dW + 
                                        0.5 * sigma**2 * (dW**2 - dt))
        
        return t, S
    
    @staticmethod
    def ornstein_uhlenbeck(X0: float, theta: float, mu: float, sigma: float,
                          T: float, steps: int, paths: int) -> Tuple[np.ndarray, np.ndarray]:
        """Ornstein-Uhlenbeck mean-reverting process"""
        dt = T / steps
        t = np.linspace(0, T, steps + 1)
        X = np.zeros((paths, steps + 1))
        X[:, 0] = X0
        
        for i in range(steps):
            dW = np.sqrt(dt) * np.random.randn(paths)
            X[:, i + 1] = X[:, i] + theta * (mu - X[:, i]) * dt + sigma * dW
        
        return t, X
    
    @staticmethod
    def heston_model(S0: float, v0: float, mu: float, kappa: float, 
                    theta: float, sigma_v: float, rho: float, 
                    T: float, steps: int, paths: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Heston stochastic volatility model"""
        dt = T / steps
        t = np.linspace(0, T, steps + 1)
        S = np.zeros((paths, steps + 1))
        v = np.zeros((paths, steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        for i in range(steps):
            # Correlated random numbers
            Z1 = np.random.randn(paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(paths)
            
            # Ensure variance stays positive (Feller condition)
            v_sqrt = np.sqrt(np.maximum(v[:, i], 0))
            
            # Asset price
            S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * v[:, i]) * dt + v_sqrt * np.sqrt(dt) * Z1)
            
            # Variance
            v[:, i + 1] = np.maximum(v[:, i] + kappa * (theta - v[:, i]) * dt + 
                                    sigma_v * v_sqrt * np.sqrt(dt) * Z2, 0)
        
        return t, S, v


class PerformanceTracker:
    """Track performance metrics for simulations"""
    
    def __init__(self):
        self.metrics = []
    
    def track(self, name: str, func: Callable, *args, **kwargs) -> Dict:
        """Track execution time and memory usage"""
        start_time = time.time()
        start_mem = self._get_memory_usage()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_mem = self._get_memory_usage()
        
        metric = {
            'name': name,
            'time': end_time - start_time,
            'memory': end_mem - start_mem,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.metrics.append(metric)
        
        return {
            'result': result,
            'metrics': metric
        }
    
    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage (placeholder)"""
        # In production, use psutil or memory_profiler
        return 0.0
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        if not self.metrics:
            return {}
        
        times = [m['time'] for m in self.metrics]
        
        return {
            'total_time': sum(times),
            'avg_time': np.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'n_operations': len(self.metrics)
        }


class GreeksCalculator:
    """Enhanced Greeks calculation with multiple methods"""
    
    @staticmethod
    def finite_difference(price_func: Callable, S: float, epsilon: float,
                         param: str = 'S') -> float:
        """Calculate Greek using central finite difference"""
        if param == 'S':
            up = price_func(S + epsilon)
            down = price_func(S - epsilon)
            return (up - down) / (2 * epsilon)
        
        return 0.0
    
    @staticmethod
    def pathwise(payoff_func: Callable, paths: np.ndarray, 
                 discount: float) -> float:
        """Pathwise derivative method for Delta"""
        # Derivative of payoff with respect to initial price
        # More efficient than finite differences
        derivatives = np.zeros(len(paths))
        
        for i, path in enumerate(paths):
            if payoff_func(path) > 0:
                derivatives[i] = 1.0  # Simplified for call option
        
        return discount * np.mean(derivatives)
    
    @staticmethod
    def likelihood_ratio(payoff_func: Callable, paths: np.ndarray,
                        discount: float, mu: float, sigma: float, 
                        T: float) -> float:
        """Likelihood ratio method (score function)"""
        payoffs = np.array([payoff_func(p) for p in paths])
        
        # Score function
        W = np.random.randn(len(paths))
        scores = W / (sigma * np.sqrt(T))
        
        return discount * np.mean(payoffs * scores)


# Utility functions for common operations
def black_scholes_formula(S: float, K: float, T: float, r: float, 
                         sigma: float, option_type: str = 'call') -> Dict:
    """
    Complete Black-Scholes pricing with Greeks
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2) / 100
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


def generate_correlated_returns(mu: np.ndarray, cov: np.ndarray, 
                               n_samples: int) -> np.ndarray:
    """
    Generate correlated returns using Cholesky decomposition
    """
    n_assets = len(mu)
    
    # Cholesky decomposition
    try:
        L = cholesky(cov, lower=True)
    except np.linalg.LinAlgError:
        # Add small regularization if matrix is not positive definite
        cov_reg = cov + np.eye(n_assets) * 1e-8
        L = cholesky(cov_reg, lower=True)
    
    # Generate independent standard normals
    Z = np.random.randn(n_samples, n_assets)
    
    # Transform to correlated returns
    returns = Z @ L.T + mu
    
    return returns


def estimate_var_cvar(returns: np.ndarray, confidence: float = 0.95) -> Dict:
    """
    Estimate Value at Risk and Conditional VaR
    """
    sorted_returns = np.sort(returns)
    var_index = int((1 - confidence) * len(returns))
    
    var = -sorted_returns[var_index] if var_index < len(returns) else 0
    cvar = -np.mean(sorted_returns[:var_index]) if var_index > 0 else var
    
    return {
        'var': var,
        'cvar': cvar,
        'confidence': confidence,
        'var_index': var_index
    }