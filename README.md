# Monte Carlo Simulation Platform for Financial Engineering

A comprehensive web-based platform implementing Monte Carlo simulation for quantitative finance, based on the Chen & Hong (2007) framework. Features derivatives pricing, risk management, and interactive financial analysis tools.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/monte-carlo-finance.git
cd monte-carlo-finance
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Access the platform**
Open your browser and navigate to: [http://localhost:5000](http://localhost:5000)

## 📋 Features

- **European Option Pricing**: Monte Carlo simulation with Black-Scholes validation
- **American Option Pricing**: Longstaff-Schwartz algorithm implementation
- **Risk Management**: VaR, CVaR, and Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- **Path Generation**: Euler, Milstein, and exact simulation schemes
- **Variance Reduction**: Antithetic variates, control variates, importance sampling
- **Portfolio Analysis**: Multi-asset correlation modeling and risk decomposition
- **Interactive Web Interface**: Real-time parameter tuning and visualization

## 🏗️ Project Structure

```
monte-carlo-finance/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── templates/                  # HTML interfaces
│   ├── index.html             # Dashboard
│   ├── european_options.html  # European pricing
│   ├── american_options.html  # American pricing
│   ├── greeks.html            # Greeks calculator
│   ├── portfolio_var.html     # Risk management
│   ├── path_generation.html   # Path simulation
│   └── variance_reduction.html # Optimization
└── utils/                      # Core algorithms
    ├── monte_carlo.py         # Monte Carlo methods
    └── financial_models.py    # Financial models
```

## 📖 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/price-european-option` | POST | Price European options |
| `/api/price-american-option` | POST | Price American options |
| `/api/calculate-greeks` | POST | Calculate option Greeks |
| `/api/generate-paths-advanced` | POST | Generate stochastic paths |
| `/api/portfolio-var-enhanced` | POST | Calculate portfolio VaR |
| `/api/variance-reduction-detailed` | POST | Compare variance techniques |

## 📊 Example Usage

1. **European Call Option**: Navigate to European Options page, enter parameters, run simulation
2. **American Put Option**: Use American Options page with Longstaff-Schwartz algorithm
3. **Portfolio Risk**: Enter portfolio weights and correlations in Portfolio VaR page
4. **Path Comparison**: Visualize Euler vs Milstein schemes in Path Generation page

## 🧪 Testing

Run basic functionality tests:
```bash
# Test European option pricing
curl -X POST http://localhost:5000/api/price-european-option \
  -H "Content-Type: application/json" \
  -d '{"S0":100,"K":105,"T":1,"r":0.05,"sigma":0.2,"simulations":10000,"optionType":"call"}'
```

## 📚 Theoretical Basis

Implements methods from:
- **Chen & Hong (2007)**: Monte Carlo Simulation in Financial Engineering
- **Black & Scholes (1973)**: Option pricing model
- **Longstaff & Schwartz (2001)**: American option pricing
- **Broadie & Glasserman (2004)**: Stochastic mesh methods



**Note**: This is an educational tool for financial simulation. Not intended for real trading or investment decisions.
