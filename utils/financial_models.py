"""
Enhanced Financial Models
Advanced option pricing, Greeks, and risk metrics
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class OptionPricer:
    """Advanced option pricing methods"""
    
    @staticmethod
    def european_monte_carlo(S0: float, K: float, T: float, r: float, 
                            sigma: float, n_sims: int, 
                            option_type: str = 'call',
                            use_antithetic: bool = False) -> Dict:
        """
        European option pricing with optional variance reduction
        """
        # Generate random numbers
        if use_antithetic:
            n_half = n_sims // 2
            Z = np.random.randn(n_half)
            Z_full = np.concatenate([Z, -Z])
        else:
            Z_full = np.random.randn(n_sims)
        
        # Simulate final stock prices
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_full)
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount to present value
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
        
        return {
            'price': price,
            'std_error': std_error,
            'payoffs': payoffs,
            'final_prices': ST,
            'min_payoff': np.min(payoffs),
            'max_payoff': np.max(payoffs)
        }
    
    @staticmethod
    def american_longstaff_schwartz(S0: float, K: float, T: float, r: float,
                                   sigma: float, steps: int, n_paths: int,
                                   option_type: str = 'put') -> Dict:
        """
        American option pricing using Longstaff-Schwartz LSM algorithm
        """
        dt = T / steps
        discount = np.exp(-r * dt)
        
        # Generate paths
        paths = np.zeros((n_paths, steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, steps + 1):
            Z = np.random.randn(n_paths)
            paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + 
                                                  sigma * np.sqrt(dt) * Z)
        
        # Calculate payoffs
        if option_type == 'call':
            payoff_matrix = np.maximum(paths - K, 0)
        else:
            payoff_matrix = np.maximum(K - paths, 0)
        
        # Initialize value matrix
        value = payoff_matrix[:, -1].copy()
        
        # Backward induction
        exercise_times = np.full(n_paths, steps)
        
        for t in range(steps - 1, 0, -1):
            # Identify in-the-money paths
            itm = payoff_matrix[:, t] > 0
            
            if np.sum(itm) == 0:
                value = value * discount
                continue
            
            # Regression on in-the-money paths
            X = paths[itm, t]
            Y = value[itm] * discount
            
            # Basis functions: 1, S, S^2
            A = np.column_stack([np.ones_like(X), X, X**2])
            
            try:
                beta = np.linalg.lstsq(A, Y, rcond=None)[0]
                continuation_value = A @ beta
            except:
                continuation_value = np.mean(Y) * np.ones_like(X)
            
            # Exercise decision
            exercise = payoff_matrix[itm, t] > continuation_value
            
            # Update values and exercise times
            value_itm = value[itm].copy()
            value_itm[exercise] = payoff_matrix[itm, t][exercise]
            value[itm] = value_itm
            
            exercise_times[itm & (payoff_matrix[:, t] > continuation_value)] = t
            
            # Discount continuation values
            value[~itm] *= discount
        
        # Calculate price
        price = np.mean(value) * discount
        
        # European price for comparison
        european_payoffs = payoff_matrix[:, -1]
        european_price = np.exp(-r * T) * np.mean(european_payoffs)
        
        # Exercise boundary approximation
        exercise_boundary = []
        for t in range(1, steps + 1):
            exercised_at_t = exercise_times == t
            if np.any(exercised_at_t):
                exercise_boundary.append({
                    'time': t * dt,
                    'mean_price': np.mean(paths[exercised_at_t, t]),
                    'count': np.sum(exercised_at_t)
                })
        
        return {
            'american_price': price,
            'european_price': european_price,
            'early_exercise_premium': price - european_price,
            'std_error': np.std(value) / np.sqrt(n_paths),
            'exercise_boundary': exercise_boundary,
            'avg_exercise_time': np.mean(exercise_times) * dt,
            'paths': paths[:min(10, n_paths), :].tolist()  # Sample paths
        }
    
    @staticmethod
    def asian_option(S0: float, K: float, T: float, r: float, sigma: float,
                    n_sims: int, n_steps: int, option_type: str = 'call',
                    avg_type: str = 'arithmetic') -> Dict:
        """
        Asian option pricing (path-dependent)
        """
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = S0
        
        # Generate paths
        for i in range(n_steps):
            Z = np.random.randn(n_sims)
            paths[:, i+1] = paths[:, i] * np.exp((r - 0.5 * sigma**2) * dt + 
                                                  sigma * np.sqrt(dt) * Z)
        
        # Calculate average
        if avg_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:  # geometric
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        
        # Payoffs
        if option_type == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)
        
        price = np.exp(-r * T) * np.mean(payoffs)
        
        return {
            'price': price,
            'std_error': np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims),
            'avg_prices': avg_prices[:100].tolist(),
            'payoffs': payoffs[:100].tolist()
        }
    
    @staticmethod
    def barrier_option(S0: float, K: float, T: float, r: float, sigma: float,
                      barrier: float, n_sims: int, n_steps: int,
                      barrier_type: str = 'down-and-out',
                      option_type: str = 'call') -> Dict:
        """
        Barrier option pricing (knock-out/knock-in)
        """
        dt = T / n_steps
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = S0
        knocked_out = np.zeros(n_sims, dtype=bool)
        
        # Generate paths and check barrier
        for i in range(n_steps):
            Z = np.random.randn(n_sims)
            paths[:, i+1] = paths[:, i] * np.exp((r - 0.5 * sigma**2) * dt + 
                                                  sigma * np.sqrt(dt) * Z)
            
            if barrier_type == 'down-and-out':
                knocked_out |= paths[:, i+1] <= barrier
            elif barrier_type == 'up-and-out':
                knocked_out |= paths[:, i+1] >= barrier
            elif barrier_type == 'down-and-in':
                knocked_out |= paths[:, i+1] > barrier  # Still alive
            elif barrier_type == 'up-and-in':
                knocked_out |= paths[:, i+1] < barrier
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(paths[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - paths[:, -1], 0)
        
        # Apply barrier condition
        if 'out' in barrier_type:
            payoffs[knocked_out] = 0
        else:  # knock-in
            payoffs[~knocked_out] = 0
        
        price = np.exp(-r * T) * np.mean(payoffs)
        knockout_prob = np.mean(knocked_out)
        
        return {
            'price': price,
            'knockout_probability': float(knockout_prob),
            'std_error': np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims),
            'payoffs': payoffs[:100].tolist()
        }


class GreeksEngine:
    """Advanced Greeks calculation"""
    
    @staticmethod
    def calculate_all_greeks(S0: float, K: float, T: float, r: float,
                           sigma: float, n_sims: int, 
                           option_type: str = 'call',
                           method: str = 'finite_difference') -> Dict:
        """
        Calculate all option Greeks using specified method
        """
        epsilon_S = S0 * 0.01  # 1% of stock price
        epsilon_sigma = 0.01  # 1% absolute
        epsilon_r = 0.0001
        epsilon_T = 1/365  # 1 day
        
        def price_func(S, K, T, r, sigma):
            Z = np.random.randn(n_sims)
            ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
            if option_type == 'call':
                payoffs = np.maximum(ST - K, 0)
            else:
                payoffs = np.maximum(K - ST, 0)
            return np.exp(-r * T) * np.mean(payoffs)
        
        # Base price
        price = price_func(S0, K, T, r, sigma)
        
        # Delta (∂V/∂S)
        price_up_S = price_func(S0 + epsilon_S, K, T, r, sigma)
        price_down_S = price_func(S0 - epsilon_S, K, T, r, sigma)
        delta = (price_up_S - price_down_S) / (2 * epsilon_S)
        
        # Gamma (∂²V/∂S²)
        gamma = (price_up_S - 2 * price + price_down_S) / (epsilon_S**2)
        
        # Vega (∂V/∂σ)
        price_up_sigma = price_func(S0, K, T, r, sigma + epsilon_sigma)
        vega = (price_up_sigma - price) / epsilon_sigma
        
        # Theta (∂V/∂T)
        if T > epsilon_T:
            price_down_T = price_func(S0, K, T - epsilon_T, r, sigma)
            theta = (price_down_T - price) / epsilon_T
        else:
            theta = 0
        
        # Rho (∂V/∂r)
        price_up_r = price_func(S0, K, T, r + epsilon_r, sigma)
        rho = (price_up_r - price) / epsilon_r
        
        # Black-Scholes Greeks for comparison
        bs_greeks = OptionPricer.black_scholes_greeks(S0, K, T, r, sigma, option_type)
        
        return {
            'monte_carlo': {
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta,
                'rho': rho
            },
            'black_scholes': bs_greeks,
            'differences': {
                'price': abs(price - bs_greeks['price']),
                'delta': abs(delta - bs_greeks['delta']),
                'gamma': abs(gamma - bs_greeks['gamma']),
                'vega': abs(vega - bs_greeks['vega']),
                'theta': abs(theta - bs_greeks['theta']),
                'rho': abs(rho - bs_greeks['rho'])
            }
        }
    
    @staticmethod
    def greek_surface(S_range: np.ndarray, sigma_range: np.ndarray,
                     K: float, T: float, r: float,
                     greek: str = 'delta') -> np.ndarray:
        """
        Calculate Greek surface for visualization
        """
        surface = np.zeros((len(S_range), len(sigma_range)))
        
        for i, S in enumerate(S_range):
            for j, sigma in enumerate(sigma_range):
                greeks = OptionPricer.black_scholes_greeks(S, K, T, r, sigma, 'call')
                surface[i, j] = greeks[greek]
        
        return surface


# Add static method to OptionPricer class
def black_scholes_greeks(S: float, K: float, T: float, r: float,
                        sigma: float, option_type: str = 'call') -> Dict:
    """Black-Scholes analytical Greeks"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

OptionPricer.black_scholes_greeks = staticmethod(black_scholes_greeks)


class RiskMetrics:
    """Risk management metrics and calculations"""
    
    @staticmethod
    def calculate_var_cvar(returns: np.ndarray, confidence: float = 0.95,
                          portfolio_value: float = 1.0) -> Dict:
        """
        Calculate Value at Risk and Conditional VaR
        """
        losses = -returns * portfolio_value
        sorted_losses = np.sort(losses)
        
        var_index = int((1 - confidence) * len(losses))
        var = sorted_losses[var_index] if var_index < len(losses) else 0
        cvar = np.mean(sorted_losses[var_index:]) if var_index < len(losses) else var
        
        return {
            'var': float(var),
            'cvar': float(cvar),
            'confidence': confidence,
            'var_percentile': (1 - confidence) * 100,
            'expected_shortfall': float(cvar),
            'loss_distribution': sorted_losses[:1000].tolist()
        }
    
    @staticmethod
    def portfolio_risk_decomposition(weights: np.ndarray, returns: np.ndarray,
                                    cov_matrix: np.ndarray) -> Dict:
        """
        Decompose portfolio risk into component contributions
        """
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Marginal contributions
        marginal_contributions = (cov_matrix @ weights) / portfolio_vol
        
        # Component contributions
        component_contributions = weights * marginal_contributions
        
        # Percentage contributions
        pct_contributions = component_contributions / portfolio_vol * 100
        
        return {
            'portfolio_volatility': float(portfolio_vol),
            'portfolio_variance': float(portfolio_variance),
            'marginal_contributions': marginal_contributions.tolist(),
            'component_contributions': component_contributions.tolist(),
            'percentage_contributions': pct_contributions.tolist(),
            'diversification_ratio': float(np.sum(weights * np.sqrt(np.diag(cov_matrix))) / portfolio_vol)
        }