from flask import Flask, render_template, request, jsonify, session
import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky
import json
import uuid
app = Flask(__name__)

# Utility functions for Monte Carlo simulations

def box_muller_transform():
    """Generate standard normal random variable using Box-Muller"""
    u1 = np.random.random()
    u2 = np.random.random()
    return np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

def black_scholes_price(S0, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pi-estimation')
def pi_estimation():
    return render_template('pi_estimation.html')

@app.route('/api/estimate-pi', methods=['POST'])
def estimate_pi():
    data = request.json
    samples = int(data.get('samples', 1000))
    
    points = []
    inside_circle = 0
    
    for i in range(samples):
        x = np.random.random() * 2 - 1
        y = np.random.random() * 2 - 1
        distance = np.sqrt(x**2 + y**2)
        
        if distance <= 1:
            inside_circle += 1
            if len(points) < 1000:  # Limit points sent to frontend
                points.append({'x': x, 'y': y, 'inside': True})
        else:
            if len(points) < 1000:
                points.append({'x': x, 'y': y, 'inside': False})
    
    pi_estimate = (inside_circle / samples) * 4
    error = abs(np.pi - pi_estimate)
    
    return jsonify({
        'estimate': pi_estimate,
        'actual': np.pi,
        'error': error,
        'insideCircle': inside_circle,
        'totalSamples': samples,
        'points': points
    })

@app.route('/european-options')
def european_options():
    return render_template('european_options.html')

@app.route('/api/price-european-option', methods=['POST'])
def price_european_option():
    data = request.json
    S0 = float(data['S0'])
    K = float(data['K'])
    T = float(data['T'])
    r = float(data['r'])
    sigma = float(data['sigma'])
    simulations = int(data['simulations'])
    option_type = data['optionType']
    
    payoffs = []
    final_prices = []
    
    for _ in range(simulations):
        z = box_muller_transform()
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        final_prices.append(ST)
        
        if option_type == 'call':
            payoff = max(ST - K, 0)
        else:
            payoff = max(K - ST, 0)
        payoffs.append(payoff)
    
    avg_payoff = np.mean(payoffs)
    mc_price = np.exp(-r * T) * avg_payoff
    std_error = np.std(payoffs) / np.sqrt(simulations)
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type)
    
    return jsonify({
        'mcPrice': mc_price,
        'bsPrice': bs_price,
        'stdError': std_error,
        'difference': abs(mc_price - bs_price),
        'payoffs': payoffs[:100],
        'finalPrices': final_prices[:100]
    })

@app.route('/american-options')
def american_options():
    return render_template('american_options.html')

@app.route('/api/price-american-option', methods=['POST'])
def price_american_option():
    data = request.json
    S0 = float(data['S0'])
    K = float(data['K'])
    T = float(data['T'])
    r = float(data['r'])
    sigma = float(data['sigma'])
    steps = int(data['steps'])
    simulations = int(data['simulations'])
    option_type = data['optionType']
    
    dt = T / steps
    
    # Generate all paths
    paths = []
    for _ in range(simulations):
        path = [S0]
        S = S0
        for _ in range(steps):
            z = box_muller_transform()
            S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            path.append(S)
        paths.append(path)
    
    # Calculate exercise values
    exercise_values = []
    for path in paths:
        if option_type == 'call':
            exercise_values.append([max(S - K, 0) for S in path])
        else:
            exercise_values.append([max(K - S, 0) for S in path])
    
    # Longstaff-Schwartz backward induction
    cashflows = [ev[-1] for ev in exercise_values]
    
    for t in range(steps - 1, 0, -1):
        in_money = []
        X = []
        Y = []
        
        for i in range(simulations):
            if exercise_values[i][t] > 0:
                in_money.append(i)
                S = paths[i][t]
                X.append([1, S, S**2])
                Y.append(cashflows[i] * np.exp(-r * dt))
        
        if len(in_money) > 0:
            X = np.array(X)
            Y = np.array(Y)
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            
            for idx, i in enumerate(in_money):
                S = paths[i][t]
                continuation = beta[0] + beta[1] * S + beta[2] * S**2
                
                if exercise_values[i][t] > continuation:
                    cashflows[i] = exercise_values[i][t]
    
    american_price = np.mean(cashflows) * np.exp(-r * dt)
    
    # European price for comparison
    european_payoffs = []
    for path in paths:
        if option_type == 'call':
            european_payoffs.append(max(path[-1] - K, 0))
        else:
            european_payoffs.append(max(K - path[-1], 0))
    
    european_price = np.exp(-r * T) * np.mean(european_payoffs)
    
    # Sample paths for visualization
    sample_paths = []
    for i in range(min(5, simulations)):
        path_data = [{'time': j, 'price': paths[i][j], 'path': i} for j in range(len(paths[i]))]
        sample_paths.append(path_data)
    
    return jsonify({
        'americanPrice': american_price,
        'europeanPrice': european_price,
        'earlyExercisePremium': american_price - european_price,
        'paths': sample_paths
    })

@app.route('/greeks')
def greeks():
    return render_template('greeks.html')

@app.route('/api/calculate-greeks', methods=['POST'])
def calculate_greeks():
    try:
        data = request.get_json()
        
        # Validate required parameters
        required_params = ['S0', 'K', 'T', 'r', 'sigma', 'simulations', 'optionType', 'epsilon']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'Missing parameter: {param}'}), 400
        
        # Extract parameters
        S0 = float(data['S0'])
        K = float(data['K'])
        T = float(data['T'])
        r = float(data['r'])
        sigma = float(data['sigma'])
        simulations = int(data['simulations'])
        option_type = data['optionType']
        epsilon = float(data['epsilon'])
        
        # Additional validation
        if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return jsonify({'error': 'All parameters must be positive'}), 400
        
        if simulations < 1000:
            return jsonify({'error': 'Minimum 1000 simulations required'}), 400
        
        # Calculate Greeks using your Monte Carlo implementation
        results = calculate_option_greeks_monte_carlo(
            S0, K, T, r, sigma, simulations, option_type, epsilon
        )
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_option_greeks_monte_carlo(S0, K, T, r, sigma, simulations, option_type, epsilon):
    """
    Calculate option Greeks using Monte Carlo with finite differences
    """
    # Your Monte Carlo implementation here
    # This should return a dictionary with: price, delta, gamma, vega, theta, rho
    
    # Placeholder implementation - replace with your actual Monte Carlo code
    import numpy as np
    from scipy.stats import norm
    
    # For now, using Black-Scholes for demonstration
    # In practice, you'd use Monte Carlo simulation
    def black_scholes(S, K, T, r, sigma, option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
        
        return price, delta
    
    # Calculate base price
    price, delta = black_scholes(S0, K, T, r, sigma, option_type)
    
    # Finite difference Greeks (central differences)
    # Delta and Gamma
    price_up, _ = black_scholes(S0 + epsilon, K, T, r, sigma, option_type)
    price_down, _ = black_scholes(S0 - epsilon, K, T, r, sigma, option_type)
    delta_fd = (price_up - price_down) / (2 * epsilon)
    gamma = (price_up - 2 * price + price_down) / (epsilon ** 2)
    
    # Vega
    price_vega_up, _ = black_scholes(S0, K, T, r, sigma + epsilon, option_type)
    vega = (price_vega_up - price) / epsilon
    
    # Theta (using 1 day time decay)
    dt = 1/365
    price_theta, _ = black_scholes(S0, K, T - dt, r, sigma, option_type)
    theta = (price_theta - price) / dt  # Daily theta
    
    # Rho
    price_rho_up, _ = black_scholes(S0, K, T, r + epsilon, sigma, option_type)
    rho = (price_rho_up - price) / epsilon
    
    return {
        'price': float(price),
        'delta': float(delta_fd),
        'gamma': float(gamma),
        'vega': float(vega),
        'theta': float(theta),
        'rho': float(rho)
    }

@app.route('/path-generation')
def path_generation():
    return render_template('path_generation.html')

@app.route('/api/generate-paths', methods=['POST'])
def generate_paths():
    data = request.json
    S0 = float(data['S0'])
    mu = float(data['mu'])
    sigma = float(data['sigma'])
    T = float(data['T'])
    steps = int(data['steps'])
    paths_count = int(data['paths'])
    scheme = data['scheme']
    
    dt = T / steps
    paths_data = []
    
    for p in range(paths_count):
        path = [{'time': 0, 'price': S0, 'path': p}]
        S = S0
        
        for i in range(1, steps + 1):
            z = box_muller_transform()
            
            if scheme == 'euler':
                S = S + mu * S * dt + sigma * S * np.sqrt(dt) * z
            else:  # milstein
                S = S + mu * S * dt + sigma * S * np.sqrt(dt) * z + \
                    0.5 * sigma**2 * S * (z**2 - 1) * dt
            
            path.append({'time': i * dt, 'price': S, 'path': p})
        
        paths_data.extend(path)
    
    return jsonify({'paths': paths_data})

@app.route('/scheme-comparison')
def scheme_comparison():
    return render_template('scheme_comparison.html')

@app.route('/api/compare-schemes', methods=['POST'])
def compare_schemes():
    data = request.json
    S0 = float(data['S0'])
    mu = float(data['mu'])
    sigma = float(data['sigma'])
    T = float(data['T'])
    steps = int(data['steps'])
    trials = int(data['trials'])
    
    dt = T / steps
    euler_errors = []
    milstein_errors = []
    euler_final_prices = []
    milstein_final_prices = []
    exact_final_prices = []
    
    for trial_idx in range(trials):
        # Generate same random numbers for both schemes for fair comparison
        z_values = np.random.normal(0, 1, steps)
        
        # Euler path
        S_euler = S0
        for z in z_values:
            dW = np.sqrt(dt) * z
            S_euler = S_euler + mu * S_euler * dt + sigma * S_euler * dW
        
        # Milstein path  
        S_milstein = S0
        for z in z_values:
            dW = np.sqrt(dt) * z
            S_milstein = S_milstein + mu * S_milstein * dt + sigma * S_milstein * dW + \
                        0.5 * sigma**2 * S_milstein * (z**2 - 1) * dt
        
        # Exact solution using same Brownian increments
        W_T = np.sum(z_values) * np.sqrt(dt)  # Total Brownian motion
        S_exact = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)
        
        # Store final prices for statistics
        euler_final_prices.append(S_euler)
        milstein_final_prices.append(S_milstein)
        exact_final_prices.append(S_exact)
        
        # Calculate percentage errors (more interpretable)
        euler_error = abs(S_euler - S_exact) / S_exact * 100  # Percentage error
        milstein_error = abs(S_milstein - S_exact) / S_exact * 100
        
        euler_errors.append(euler_error)
        milstein_errors.append(milstein_error)
    
    # Calculate metrics using percentage errors
    euler_mae = np.mean(euler_errors)
    milstein_mae = np.mean(milstein_errors)
    euler_rmse = np.sqrt(np.mean(np.array(euler_errors)**2))
    milstein_rmse = np.sqrt(np.mean(np.array(milstein_errors)**2))
    
    improvement = round(((euler_mae - milstein_mae) / euler_mae * 100), 2) if euler_mae > 0 else 0    
    # Additional statistics
    euler_bias = np.mean(np.array(euler_final_prices) - np.array(exact_final_prices)) / S0 * 100
    milstein_bias = np.mean(np.array(milstein_final_prices) - np.array(exact_final_prices)) / S0 * 100
    
    # Prepare error data for plotting
    euler_plot_errors = [{'trial': i, 'error': e} for i, e in enumerate(euler_errors[:100])]
    milstein_plot_errors = [{'trial': i, 'error': e} for i, e in enumerate(milstein_errors[:100])]
    
    return jsonify({
        'eulerMAE': euler_mae,
        'milsteinMAE': milstein_mae,
        'eulerRMSE': euler_rmse,
        'milsteinRMSE': milstein_rmse,
        'improvement': improvement,
        'eulerBias': euler_bias,
        'milsteinBias': milstein_bias,
        'eulerErrors': euler_plot_errors,
        'milsteinErrors': milstein_plot_errors,
        'exactPrice': np.mean(exact_final_prices)
    })
@app.route('/var')
def var():
    return render_template('var.html')

@app.route('/api/calculate-var', methods=['POST'])
def calculate_var():
    data = request.json
    portfolio_value = float(data['portfolioValue'])
    mu = float(data['mu'])
    sigma = float(data['sigma'])
    days = int(data['days'])
    confidence = float(data['confidence'])
    simulations = int(data['simulations'])
    
    dt = days / 252
    losses = []
    
    for _ in range(simulations):
        z = box_muller_transform()
        return_ = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        final_value = portfolio_value * np.exp(return_)
        loss = portfolio_value - final_value
        losses.append(loss)
    
    losses.sort(reverse=True)
    var_index = int((1 - confidence) * simulations)
    var_value = losses[var_index]
    cvar_value = np.mean(losses[:var_index]) if var_index > 0 else var_value
    
    return jsonify({
        'var': var_value,
        'cvar': cvar_value,
        'losses': losses[:100],
        'confidence': confidence * 100
    })
# Add to app.py
@app.route('/asian-options')
def asian_options():
    return render_template('asian_options.html')

@app.route('/api/price-asian-option', methods=['POST'])
def price_asian_option():
    data = request.json
    S0 = float(data['S0'])
    K = float(data['K'])
    T = float(data['T'])
    r = float(data['r'])
    sigma = float(data['sigma'])
    simulations = int(data['simulations'])
    option_type = data['optionType']
    avg_type = data['avgType']  # arithmetic or geometric
    
    dt = T / 252  # daily averaging
    steps = int(T * 252)
    
    payoffs = []
    
    for _ in range(simulations):
        path = [S0]
        S = S0
        for _ in range(steps):
            z = box_muller_transform()
            S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            path.append(S)
        
        if avg_type == 'arithmetic':
            avg_price = np.mean(path)
        else:  # geometric
            avg_price = np.exp(np.mean(np.log(path)))
        
        if option_type == 'call':
            payoff = max(avg_price - K, 0)
        else:
            payoff = max(K - avg_price, 0)
        
        payoffs.append(payoff)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return jsonify({
        'price': price,
        'stdError': np.std(payoffs) / np.sqrt(simulations),
        'payoffs': payoffs[:100]
    })

@app.route('/barrier-options')
def barrier_options():
    return render_template('barrier_options.html')

@app.route('/api/price-barrier-option', methods=['POST'])
def price_barrier_option():
    data = request.json
    S0 = float(data['S0'])
    K = float(data['K'])
    T = float(data['T'])
    r = float(data['r'])
    sigma = float(data['sigma'])
    barrier = float(data['barrier'])
    option_type = data['optionType']  # up-and-out, down-and-out, etc.
    simulations = int(data['simulations'])
    
    dt = T / 100
    steps = 100
    
    payoffs = []
    
    for _ in range(simulations):
        path = [S0]
        S = S0
        knocked_out = False
        
        for _ in range(steps):
            z = box_muller_transform()
            S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            path.append(S)
            
            # Check barrier condition
            if option_type == 'up-and-out' and S >= barrier:
                knocked_out = True
                break
            elif option_type == 'down-and-out' and S <= barrier:
                knocked_out = True
                break
        
        if knocked_out:
            payoff = 0
        else:
            if 'call' in option_type:
                payoff = max(S - K, 0)
            else:
                payoff = max(K - S, 0)
        
        payoffs.append(payoff)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return jsonify({
        'price': price,
        'knockOutProbability': payoffs.count(0) / simulations,
        'payoffs': payoffs[:100]
    })

@app.route('/portfolio-var')
def portfolio_var():
    return render_template('portfolio_var.html')

@app.route('/api/calculate-portfolio-var', methods=['POST'])
def calculate_portfolio_var():
    try:
        data = request.json
        
        # Extract parameters
        portfolio_weights = np.array(data['weights'])
        asset_params = data['returns']     
        asset_correlations = np.array(data['correlations'])
        portfolio_value = float(data['portfolioValue'])
        confidence = float(data['confidence'])
        simulations = int(data['simulations'])
        time_horizon_days = int(data.get('days', 10))
        
        n_assets = len(portfolio_weights)
        
        # Extract annual parameters (already in decimal form from frontend)
        expected_returns_annual = np.array([param[0] for param in asset_params])
        volatilities_annual = np.array([param[1] for param in asset_params])
        
        # Convert to period parameters
        trading_days_per_year = 252
        time_horizon_years = time_horizon_days / trading_days_per_year
        
        # Period parameters for the entire horizon
        expected_returns_period = expected_returns_annual * time_horizon_years
        volatilities_period = volatilities_annual * np.sqrt(time_horizon_years)
        
        # Build covariance matrix for period returns
        cov_matrix = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(n_assets):
                cov_matrix[i,j] = asset_correlations[i][j] * volatilities_period[i] * volatilities_period[j]
        
        # Cholesky decomposition with regularization
        try:
            L = cholesky(cov_matrix, lower=True)
        except np.linalg.LinAlgError:
            cov_matrix += np.eye(n_assets) * 1e-8
            L = cholesky(cov_matrix, lower=True)
        
        # Monte Carlo simulation
        portfolio_returns = []
        
        for _ in range(simulations):
            # Generate independent standard normal variables
            z = np.random.normal(0, 1, n_assets)
            
            # Transform to correlated random variables
            correlated_shocks = L @ z
            
            # Simulate asset returns using Geometric Brownian Motion
            asset_returns = []
            for i in range(n_assets):
                # GBM formula for period return (log return)
                period_return = (expected_returns_period[i] - 0.5 * volatilities_period[i]**2) + volatilities_period[i] * correlated_shocks[i]
                asset_returns.append(period_return)
            
            # Calculate portfolio return (weighted average of asset returns)
            portfolio_return = np.sum(portfolio_weights * asset_returns)
            portfolio_returns.append(portfolio_return)
        
        # Convert to numpy array
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate portfolio final values
        portfolio_values = portfolio_value * np.exp(portfolio_returns)
        
        # FIX: Calculate PROFITS AND LOSSES (P&L) instead of just losses
        # P&L = final_value - initial_value (positive = profit, negative = loss)
        pnl = portfolio_values - portfolio_value
        
        # Sort P&L to find VaR and CVaR (we want the worst losses, which are the most negative values)
        pnl_sorted = np.sort(pnl)
        
        # VaR is the LOSS at the confidence quantile 
        # Since pnl_sorted is sorted from worst (most negative) to best (most positive)
        # The VaR is at the (1-confidence) percentile from the left
        var_index = int((1 - confidence) * simulations)
        
        # VaR is the negative of the P&L at the var_index (because VaR is expressed as a positive loss)
        var = max(0, -pnl_sorted[var_index])
        
        # Conditional VaR (average of losses beyond VaR)
        tail_losses = pnl_sorted[:var_index]  # All values from start to var_index are losses worse than VaR
        cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else var
        cvar = max(cvar, var)  # Ensure CVaR >= VaR
        
        # Calculate portfolio statistics
        portfolio_expected_return_annual = np.sum(portfolio_weights * expected_returns_annual)
        
        # Portfolio volatility calculation
        portfolio_variance_annual = 0
        for i in range(n_assets):
            for j in range(n_assets):
                portfolio_variance_annual += (portfolio_weights[i] * portfolio_weights[j] * 
                                           volatilities_annual[i] * volatilities_annual[j] * 
                                           asset_correlations[i][j])
        portfolio_volatility_annual = np.sqrt(portfolio_variance_annual)
        
        # Sharpe ratio
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_expected_return_annual - risk_free_rate) / portfolio_volatility_annual
        
        # Component VaR analysis
        component_var = []
        for i in range(n_assets):
            # Calculate marginal contribution to risk
            marginal_contribution = (
                portfolio_weights[i] * volatilities_annual[i] * 
                np.sum([portfolio_weights[j] * volatilities_annual[j] * asset_correlations[i][j] 
                       for j in range(n_assets)])
            ) / portfolio_volatility_annual
            
            component_var.append({
                'asset': i + 1,
                'weight': float(portfolio_weights[i]),
                'marginal_contribution': float(marginal_contribution),
                'component_var': float(var * marginal_contribution)
            })
        
        return jsonify({
            'var': float(var),
            'cvar': float(cvar),
            'expected_return_annual': float(portfolio_expected_return_annual),
            'volatility_annual': float(portfolio_volatility_annual),
            'sharpe_ratio': float(sharpe_ratio),
            'portfolio_returns': portfolio_returns.tolist()[:1000],  # Send simple returns for chart
            'component_var': component_var,
            'success': True
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f"{str(e)}\n{traceback.format_exc()}",
            'success': False
        }), 400

@app.route('/variance-reduction')
def variance_reduction():
    return render_template('variance_reduction.html')

@app.route('/api/compare-variance-methods', methods=['POST'])
def compare_variance_methods():
    data = request.json
    S0 = float(data['S0'])
    K = float(data['K'])
    T = float(data['T'])
    r = float(data['r'])
    sigma = float(data['sigma'])
    simulations = int(data['simulations'])
    
    # Standard Monte Carlo
    standard_prices = []
    for _ in range(100):  # Multiple runs to measure variance
        payoffs = []
        for _ in range(simulations):
            z = box_muller_transform()
            ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
            payoff = max(ST - K, 0)
            payoffs.append(payoff)
        standard_prices.append(np.exp(-r * T) * np.mean(payoffs))
    
    # Antithetic Variates
    antithetic_prices = []
    for _ in range(100):
        payoffs = []
        for _ in range(simulations // 2):
            z = box_muller_transform()
            # Use z and -z
            ST1 = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
            ST2 = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * -z)
            payoff1 = max(ST1 - K, 0)
            payoff2 = max(ST2 - K, 0)
            payoffs.extend([payoff1, payoff2])
        antithetic_prices.append(np.exp(-r * T) * np.mean(payoffs))
    
    # Control Variates
    control_prices = []
    for _ in range(100):
        payoffs = []
        stock_prices = []
        for _ in range(simulations):
            z = box_muller_transform()
            ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
            payoff = max(ST - K, 0)
            payoffs.append(payoff)
            stock_prices.append(ST)
        
        # Use stock price as control variate
        cov = np.cov(payoffs, stock_prices)[0,1]
        var_stock = np.var(stock_prices)
        theta = cov / var_stock
        
        adjusted_payoffs = [p - theta * (s - S0 * np.exp(r * T)) for p, s in zip(payoffs, stock_prices)]
        control_prices.append(np.exp(-r * T) * np.mean(adjusted_payoffs))
    
    return jsonify({
        'standard': {
            'mean': np.mean(standard_prices),
            'std': np.std(standard_prices),
            'variance': np.var(standard_prices)
        },
        'antithetic': {
            'mean': np.mean(antithetic_prices),
            'std': np.std(antithetic_prices),
            'variance': np.var(antithetic_prices)
        },
        'control': {
            'mean': np.mean(control_prices),
            'std': np.std(control_prices),
            'variance': np.var(control_prices)
        }
    })

@app.route('/convergence-analysis')
def convergence_analysis():
    return render_template('convergence_analysis.html')

@app.route('/api/analyze-convergence', methods=['POST'])
def analyze_convergence():
    data = request.json
    S0 = float(data['S0'])
    K = float(data['K'])
    T = float(data['T'])
    r = float(data['r'])
    sigma = float(data['sigma'])
    max_simulations = int(data['maxSimulations'])
    option_type = data.get('optionType', 'call')  # Get option type from request
    
    # Generate logarithmic range of simulation sizes
    simulation_sizes = []
    n = 100
    while n <= max_simulations:
        simulation_sizes.append(n)
        n = int(n * 2)  # Double each time for logarithmic spacing
    
    # Ensure we include the max_simulations
    if max_simulations not in simulation_sizes:
        simulation_sizes.append(max_simulations)
    
    # Calculate Black-Scholes price for the correct option type
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type)
    
    results = []
    
    for n in simulation_sizes:
        errors = []
        mc_prices = []
        
        # Run multiple trials for better statistics
        trials = min(20, max(5, 100000 // n))  # More trials for smaller n
        
        for trial in range(trials):
            payoffs = []
            for _ in range(n):
                z = box_muller_transform()
                ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
                
                if option_type == 'call':
                    payoff = max(ST - K, 0)
                else:  # put option
                    payoff = max(K - ST, 0)
                    
                payoffs.append(payoff)
            
            mc_price = np.exp(-r * T) * np.mean(payoffs)
            mc_prices.append(mc_price)
            error = abs(mc_price - bs_price)
            errors.append(error)
        
        # Calculate statistics
        mean_error = np.mean(errors)
        std_error = np.std(mc_prices) / np.sqrt(trials)  # Standard error of the mean
        theoretical_std = 1.0 / np.sqrt(n)  # Theoretical convergence rate
        
        results.append({
            'simulations': n,
            'meanError': mean_error,
            'stdError': std_error,
            'convergenceRate': theoretical_std * bs_price,  # Scale by price for meaningful comparison
            'mcPrice': np.mean(mc_prices),
            'trials': trials
        })
    
    return jsonify({
        'results': results, 
        'bsPrice': bs_price,
        'optionType': option_type
    })
@app.route('/api/export-results', methods=['POST'])
def export_results():
    data = request.json
    results = data['results']
    format_type = data['format']  # 'csv' or 'json'
    
    if format_type == 'csv':
        # Convert to CSV
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        if 'paths' in results:
            writer.writerow(['Time', 'Price', 'Path'])
            for path in results['paths']:
                for point in path:
                    writer.writerow([point['time'], point['price'], point['path']])
        
        return jsonify({'csv': output.getvalue()})
    
    else:  # JSON
        return jsonify({'json': json.dumps(results, indent=2)})
    
@app.route('/api/start-large-simulation', methods=['POST'])
def start_large_simulation():
    simulation_id = str(uuid.uuid4())
    session[simulation_id] = {
        'status': 'running',
        'progress': 0,
        'results': None
    }
    return jsonify({'simulationId': simulation_id})

@app.route('/api/check-simulation-status/<simulation_id>')
def check_simulation_status(simulation_id):
    return jsonify(session.get(simulation_id, {'status': 'not_found'}))


if __name__ == '__main__':
    app.run(debug=True, port=5000)