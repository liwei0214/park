#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Green-Blue Infrastructure (GBI) Environmental Performance Analysis
================================================================================
GAMM + DEEP LEARNING INTEGRATED VERSION

Key Methods:
1. GAMM (Generalized Additive Mixed Models) - using pygam & statsmodels
2. Neural Network for nonlinear relationship modeling
3. LSTM for temporal dynamics and city trajectory prediction
4. Autoencoder for feature extraction
5. Deep Learning-based threshold detection

Author: GBI Research Analysis - GAMM & Deep Learning Enhanced
Date: 2024
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# DEEP LEARNING IMPORTS
# ============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
    print("✓ PyTorch loaded successfully")
except ImportError:
    TORCH_AVAILABLE = False
    print("✗ PyTorch not available - using numpy fallback")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
    print("✓ Scikit-learn loaded successfully")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("✗ Scikit-learn not available")

# ============================================================================
# GAMM IMPORTS
# ============================================================================
try:
    from pygam import LinearGAM, GAM, s, f, te, l
    from pygam import GammaGAM, PoissonGAM

    PYGAM_AVAILABLE = True
    print("✓ PyGAM loaded successfully")
except ImportError:
    PYGAM_AVAILABLE = False
    print("✗ PyGAM not available - install with: pip install pygam")

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM

    STATSMODELS_AVAILABLE = True
    print("✓ Statsmodels loaded successfully")
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("✗ Statsmodels not available")

# ============================================================================
# CONFIGURATION
# ============================================================================
# NOTE: Update these paths to match your local data directory structure
DATA_ROOT = Path("./data")  # Change to your data directory

PATHS = {
    'co2': DATA_ROOT / "co2/city_co2_emissions.xlsx",
    'pm25': DATA_ROOT / "pm25/city_pm25_annual.xlsx",
    'population': DATA_ROOT / "population/city_population.xlsx",
    'viirs': DATA_ROOT / "viirs/city_nightlight.xlsx",
    'osm_parks': DATA_ROOT / "parks",
    'clcd': DATA_ROOT / "landcover",
}

MERGED_DATA_PATH = Path("output/merged_data.csv")

OUTPUT_DIR = Path("GBI_GAMM_DL_Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_START = 2014
YEAR_END = 2023

# Target cities configuration
CITIES = {
    'Beijing': {'cn': '北京市', 'cn_short': '北京', 'en': 'Beijing', 'is_municipality': True,
                'province': '北京', 'bbox': [115.4, 39.4, 117.5, 41.1]},
    'Shanghai': {'cn': '上海市', 'cn_short': '上海', 'en': 'Shanghai', 'is_municipality': True,
                 'province': '上海', 'bbox': [120.8, 30.7, 122.2, 31.9]},
    'Guangzhou': {'cn': '广州市', 'cn_short': '广州', 'en': 'Guangzhou', 'is_municipality': False,
                  'province': '广东', 'bbox': [112.9, 22.5, 114.1, 23.9]},
    'Shenzhen': {'cn': '深圳市', 'cn_short': '深圳', 'en': 'Shenzhen', 'is_municipality': False,
                 'province': '广东', 'bbox': [113.7, 22.4, 114.6, 22.9]},
    'Nanjing': {'cn': '南京市', 'cn_short': '南京', 'en': 'Nanjing', 'is_municipality': False,
                'province': '江苏', 'bbox': [118.3, 31.2, 119.3, 32.6]},
    'Wuhan': {'cn': '武汉市', 'cn_short': '武汉', 'en': 'Wuhan', 'is_municipality': False,
              'province': '湖北', 'bbox': [113.7, 29.9, 115.1, 31.4]},
    'Chengdu': {'cn': '成都市', 'cn_short': '成都', 'en': 'Chengdu', 'is_municipality': False,
                'province': '四川', 'bbox': [102.9, 30.0, 105.0, 31.5]},
    'Xian': {'cn': '西安市', 'cn_short': '西安', 'en': "Xi'an", 'is_municipality': False,
             'province': '陕西', 'bbox': [107.6, 33.5, 109.8, 34.8]},
    'Chongqing': {'cn': '重庆市', 'cn_short': '重庆', 'en': 'Chongqing', 'is_municipality': True,
                  'province': '重庆', 'bbox': [105.2, 28.1, 110.2, 32.2]},
}

# Matplotlib settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# Color scheme
COLORS = {
    'high_dev': '#E74C3C', 'low_dev': '#3498DB',
    'green': '#27AE60', 'blue': '#2980B9',
    'park': '#F39C12', 'forest': '#1E8449', 'water': '#1ABC9C',
    'q1': '#3498DB', 'q2': '#F1C40F', 'q3': '#E67E22', 'q4': '#E74C3C',
    'threshold': '#2ECC71',
    'High_Dev': '#E74C3C', 'Low_Dev': '#3498DB',
    'Q1': '#2ECC71', 'Q2': '#F1C40F', 'Q3': '#E67E22', 'Q4': '#E74C3C',
    'Developing': '#3498DB', 'Mature': '#E74C3C',
    'neural': '#9B59B6',
    'lstm': '#1ABC9C',
    'gamm': '#E91E63',
    'gam_smooth': '#FF5722',
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def print_header(title):
    print("\n" + "=" * 70)
    print(f"[{title}]")
    print("=" * 70)


def print_status(message, success=True):
    symbol = "✓" if success else "✗"
    print(f"  {symbol} {message}")


def compute_correlations(df, x_var, y_var):
    valid = df[[x_var, y_var]].dropna()
    if len(valid) < 5:
        return np.nan, np.nan, ''
    r, p = stats.pearsonr(valid[x_var], valid[y_var])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    return r, p, sig


# ============================================================================
# GAMM ANALYZER CLASS (NEW!)
# ============================================================================
class GAMMAnalyzer:
    """
    Generalized Additive Mixed Models Analyzer
    Implements true GAMM using pygam and statsmodels
    """

    def __init__(self):
        self.models = {}
        self.results = {}
        self.predictions = {}

    def fit_gam_pygam(self, X, y, name='default', n_splines=20, lam=0.6):
        """
        Fit GAM using pygam library

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Predictor variables
        y : array-like, shape (n_samples,)
            Response variable
        n_splines : int
            Number of splines for smooth terms
        lam : float
            Smoothing parameter (regularization)
        """
        if not PYGAM_AVAILABLE:
            print_status("PyGAM not available", False)
            return None

        try:
            X = np.array(X)
            y = np.array(y)

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # Build GAM with smooth terms for each predictor
            n_features = X.shape[1]

            if n_features == 1:
                # Single predictor: simple smooth
                gam = LinearGAM(s(0, n_splines=n_splines, lam=lam))
            elif n_features == 2:
                # Two predictors: smooth + smooth + tensor interaction
                gam = LinearGAM(s(0, n_splines=n_splines, lam=lam) +
                                s(1, n_splines=n_splines, lam=lam) +
                                te(0, 1, n_splines=n_splines // 2))
            else:
                # Multiple predictors: smooth for each
                terms = s(0, n_splines=n_splines, lam=lam)
                for i in range(1, n_features):
                    terms += s(i, n_splines=n_splines, lam=lam)
                gam = LinearGAM(terms)

            # Fit the model
            gam.gridsearch(X, y, lam=np.logspace(-3, 3, 11))

            self.models[f'gam_{name}'] = gam

            # Store results
            self.results[f'gam_{name}'] = {
                'pseudo_r2': gam.statistics_['pseudo_r2']['explained_deviance'],
                'aic': gam.statistics_['AIC'],
                'n_samples': len(y),
                'n_features': n_features,
                'edof': gam.statistics_['edof'],  # Effective degrees of freedom
            }

            print_status(f"GAM '{name}' fitted: R² = {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")
            return gam

        except Exception as e:
            print_status(f"GAM fitting failed: {e}", False)
            return None

    def fit_gamm_statsmodels(self, df, formula, groups, name='default', re_formula=None):
        """
        Fit GAMM using statsmodels Mixed Linear Model
        This implements the 'mixed' part of GAMM

        Parameters:
        -----------
        df : DataFrame
            Data containing all variables
        formula : str
            Model formula (e.g., "PM25 ~ Park_km2 + VIIRS")
        groups : str
            Column name for random effects grouping (e.g., "City")
        re_formula : str, optional
            Random effects formula
        """
        if not STATSMODELS_AVAILABLE:
            print_status("Statsmodels not available", False)
            return None

        try:
            # Fit Mixed Linear Model
            if re_formula:
                model = smf.mixedlm(formula, df, groups=df[groups], re_formula=re_formula)
            else:
                model = smf.mixedlm(formula, df, groups=df[groups])

            result = model.fit(method='lbfgs', maxiter=1000)

            self.models[f'gamm_{name}'] = result
            self.results[f'gamm_{name}'] = {
                'aic': result.aic,
                'bic': result.bic,
                'llf': result.llf,
                'converged': result.converged,
                'fe_params': result.fe_params.to_dict(),
                'random_effects': {str(k): dict(v) for k, v in result.random_effects.items()},
                'pvalues': result.pvalues.to_dict() if hasattr(result, 'pvalues') else {},
            }

            print_status(f"GAMM '{name}' fitted: AIC = {result.aic:.2f}, Converged = {result.converged}")
            return result

        except Exception as e:
            print_status(f"GAMM fitting failed: {e}", False)
            return None

    def fit_gam_with_interaction(self, df, x_var, y_var, moderator_var, name='interaction'):
        """
        Fit GAM with interaction/moderation effect
        Models: y ~ s(x) + s(moderator) + ti(x, moderator)
        """
        if not PYGAM_AVAILABLE:
            return None

        try:
            valid = df[[x_var, y_var, moderator_var]].dropna()
            X = valid[[x_var, moderator_var]].values
            y = valid[y_var].values

            # GAM with tensor interaction
            gam = LinearGAM(s(0, n_splines=15) + s(1, n_splines=15) + te(0, 1, n_splines=10))
            gam.gridsearch(X, y)

            self.models[f'gam_interaction_{name}'] = gam
            self.results[f'gam_interaction_{name}'] = {
                'pseudo_r2': gam.statistics_['pseudo_r2']['explained_deviance'],
                'aic': gam.statistics_['AIC'],
                'variables': [x_var, moderator_var, y_var],
            }

            print_status(f"GAM Interaction '{name}': R² = {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")
            return gam

        except Exception as e:
            print_status(f"GAM interaction fitting failed: {e}", False)
            return None

    def predict_gam(self, X, name='default'):
        """Generate predictions from fitted GAM"""
        model_key = f'gam_{name}'
        if model_key not in self.models:
            return None

        gam = self.models[model_key]
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = gam.predict(X)
        confidence_intervals = gam.confidence_intervals(X, width=0.95)

        return {
            'predictions': predictions,
            'ci_lower': confidence_intervals[:, 0],
            'ci_upper': confidence_intervals[:, 1],
        }

    def get_partial_dependence(self, name='default', feature_idx=0, n_points=100):
        """
        Get partial dependence plot data for a feature
        Shows the marginal effect of a predictor
        """
        model_key = f'gam_{name}'
        if model_key not in self.models:
            return None

        gam = self.models[model_key]

        try:
            XX = gam.generate_X_grid(term=feature_idx, n=n_points)
            pdep = gam.partial_dependence(term=feature_idx, X=XX)
            ci = gam.confidence_intervals(XX, width=0.95)

            return {
                'x': XX[:, feature_idx],
                'y': pdep,
                'ci_lower': ci[:, 0],
                'ci_upper': ci[:, 1],
            }
        except Exception as e:
            print_status(f"Partial dependence failed: {e}", False)
            return None

    def compare_gam_vs_linear(self, X, y, name='comparison'):
        """
        Compare GAM vs Linear model to test for nonlinearity
        """
        if not PYGAM_AVAILABLE or not STATSMODELS_AVAILABLE:
            return None

        X = np.array(X).reshape(-1, 1) if np.array(X).ndim == 1 else np.array(X)
        y = np.array(y)

        results = {}

        # Fit Linear Model
        X_with_const = sm.add_constant(X)
        linear_model = sm.OLS(y, X_with_const).fit()
        results['linear'] = {
            'r2': linear_model.rsquared,
            'aic': linear_model.aic,
            'bic': linear_model.bic,
        }

        # Fit GAM
        gam = LinearGAM(s(0, n_splines=20)).gridsearch(X, y)
        results['gam'] = {
            'r2': gam.statistics_['pseudo_r2']['explained_deviance'],
            'aic': gam.statistics_['AIC'],
        }

        # Compare
        results['improvement'] = {
            'r2_gain': results['gam']['r2'] - results['linear']['r2'],
            'aic_reduction': results['linear']['aic'] - results['gam']['aic'],
            'nonlinearity_significant': results['gam']['aic'] < results['linear']['aic'] - 2,
        }

        self.results[f'comparison_{name}'] = results
        return results

    def fit_threshold_gam(self, X, y, name='threshold'):
        """
        Fit GAM and detect threshold/changepoint using derivative analysis
        """
        if not PYGAM_AVAILABLE:
            return None

        X = np.array(X).reshape(-1, 1)
        y = np.array(y)

        # Fit smooth GAM
        gam = LinearGAM(s(0, n_splines=25, lam=0.1))
        gam.gridsearch(X, y)

        # Generate prediction grid
        x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_pred = gam.predict(x_grid)

        # Compute first derivative (slope)
        dx = x_grid[1, 0] - x_grid[0, 0]
        dy = np.diff(y_pred)
        first_derivative = dy / dx

        # Compute second derivative (curvature)
        second_derivative = np.diff(first_derivative) / dx

        # Find maximum curvature point (threshold)
        threshold_idx = np.argmax(np.abs(second_derivative)) + 1
        threshold_value = x_grid[threshold_idx, 0]

        self.models[f'gam_threshold_{name}'] = gam
        self.results[f'gam_threshold_{name}'] = {
            'threshold': threshold_value,
            'x_grid': x_grid.flatten(),
            'y_pred': y_pred,
            'first_derivative': first_derivative,
            'second_derivative': second_derivative,
            'pseudo_r2': gam.statistics_['pseudo_r2']['explained_deviance'],
        }

        print_status(f"GAM Threshold detected: {threshold_value:.2f}")
        return self.results[f'gam_threshold_{name}']


# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================
if TORCH_AVAILABLE:

    class NonlinearRegressionNet(nn.Module):
        def __init__(self, input_dim=1, hidden_dims=[64, 128, 64, 32], output_dim=1, dropout=0.2):
            super(NonlinearRegressionNet, self).__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    class LSTMTrajectoryModel(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
            super(LSTMTrajectoryModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softmax(dim=1)
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, output_dim)
            )

        def forward(self, x):
            lstm_out, (h_n, c_n) = self.lstm(x)
            attn_weights = self.attention(lstm_out)
            context = torch.sum(attn_weights * lstm_out, dim=1)
            output = self.fc(context)
            return output, attn_weights


    class AutoencoderFeatureExtractor(nn.Module):
        def __init__(self, input_dim=10, latent_dim=4, hidden_dims=[32, 16]):
            super(AutoencoderFeatureExtractor, self).__init__()
            encoder_layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                ])
                prev_dim = h_dim
            encoder_layers.append(nn.Linear(prev_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            decoder_layers = []
            prev_dim = latent_dim
            for h_dim in reversed(hidden_dims):
                decoder_layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                ])
                prev_dim = h_dim
            decoder_layers.append(nn.Linear(prev_dim, input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z


# ============================================================================
# DEEP LEARNING ANALYZER CLASS
# ============================================================================
class DeepLearningAnalyzer:
    def __init__(self, device=None):
        if TORCH_AVAILABLE:
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None
        self.models = {}
        self.scalers = {}
        self.training_history = {}

    def train_nonlinear_model(self, X, y, name='default', epochs=500, lr=0.001, verbose=False):
        if not TORCH_AVAILABLE:
            return self._fallback_nonlinear(X, y)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_x.fit_transform(X.reshape(-1, 1) if X.ndim == 1 else X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1) if y.ndim == 1 else y)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_dataset)), shuffle=True)

        input_dim = X_scaled.shape[1] if X_scaled.ndim > 1 else 1
        model = NonlinearRegressionNet(input_dim=input_dim).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}
        best_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            with torch.no_grad():
                val_x, val_y = val_dataset[:]
                val_pred = model(val_x)
                val_loss = criterion(val_pred, val_y).item()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict().copy()

        model.load_state_dict(best_state)
        self.models[name] = model
        self.scalers[name] = {'x': scaler_x, 'y': scaler_y}
        self.training_history[name] = history
        return model

    def predict_nonlinear(self, X, name='default'):
        if not TORCH_AVAILABLE or name not in self.models:
            return None

        model = self.models[name]
        scaler_x = self.scalers[name]['x']
        scaler_y = self.scalers[name]['y']

        X_scaled = scaler_x.transform(X.reshape(-1, 1) if X.ndim == 1 else X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        model.eval()
        with torch.no_grad():
            y_scaled = model(X_tensor).cpu().numpy()

        y_pred = scaler_y.inverse_transform(y_scaled)
        return y_pred.flatten()

    def detect_threshold_dl(self, X, y, name='threshold'):
        if not TORCH_AVAILABLE:
            return self._fallback_threshold(X, y)

        self.train_nonlinear_model(X, y, name=f'{name}_base', epochs=300)

        scaler_x = self.scalers[f'{name}_base']['x']
        model = self.models[f'{name}_base']

        gradients = []
        x_points = np.linspace(X.min(), X.max(), 100)

        for x_val in x_points:
            x_scaled = scaler_x.transform([[x_val]])
            x_tensor = torch.FloatTensor(x_scaled).to(self.device)
            x_tensor.requires_grad_(True)

            model.eval()
            y_pred = model(x_tensor)
            y_pred.backward()

            grad = x_tensor.grad.item()
            gradients.append(grad)

        gradients = np.array(gradients)
        grad_changes = np.abs(np.diff(gradients))
        threshold_idx = np.argmax(grad_changes)
        optimal_threshold = x_points[threshold_idx]

        return {
            'threshold': optimal_threshold,
            'x_points': x_points,
            'gradients': gradients,
            'grad_changes': grad_changes
        }

    def train_lstm_trajectories(self, df, target_col, feature_cols, seq_length=5, epochs=200):
        if not TORCH_AVAILABLE:
            return None

        sequences = []
        targets = []

        for city in df['City'].unique():
            city_data = df[df['City'] == city].sort_values('Year')
            if len(city_data) < seq_length + 1:
                continue

            features = city_data[feature_cols].values
            target = city_data[target_col].values

            for i in range(len(city_data) - seq_length):
                sequences.append(features[i:i + seq_length])
                targets.append(target[i + seq_length])

        if not sequences:
            return None

        X = np.array(sequences)
        y = np.array(targets)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled_flat = scaler_x.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X.shape)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        model = LSTMTrajectoryModel(input_dim=len(feature_cols), hidden_dim=64, num_layers=2).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output, _ = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        self.models['lstm'] = model
        self.scalers['lstm'] = {'x': scaler_x, 'y': scaler_y, 'seq_length': seq_length}
        return model

    def train_autoencoder(self, df, feature_cols, latent_dim=4, epochs=200):
        if not TORCH_AVAILABLE:
            return None

        X = df[feature_cols].dropna().values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        model = AutoencoderFeatureExtractor(input_dim=len(feature_cols), latent_dim=latent_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for batch_x, _ in loader:
                optimizer.zero_grad()
                recon, _ = model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        self.models['autoencoder'] = model
        self.scalers['autoencoder'] = scaler
        return model

    def get_latent_features(self, df, feature_cols):
        if 'autoencoder' not in self.models:
            return None
        X = df[feature_cols].dropna().values
        X_scaled = self.scalers['autoencoder'].transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.models['autoencoder'].eval()
        with torch.no_grad():
            _, latent = self.models['autoencoder'](X_tensor)
        return latent.cpu().numpy()

    def _fallback_nonlinear(self, X, y):
        return np.polyfit(X, y, 3)

    def _fallback_threshold(self, X, y):
        best_ssr = float('inf')
        best_threshold = None
        thresholds = np.linspace(np.percentile(X, 15), np.percentile(X, 85), 50)
        for t in thresholds:
            mask_below = X <= t
            mask_above = X > t
            if np.sum(mask_below) < 5 or np.sum(mask_above) < 5:
                continue
            z1 = np.polyfit(X[mask_below], y[mask_below], 1)
            z2 = np.polyfit(X[mask_above], y[mask_above], 1)
            ssr = (np.sum((y[mask_below] - np.polyval(z1, X[mask_below])) ** 2) +
                   np.sum((y[mask_above] - np.polyval(z2, X[mask_above])) ** 2))
            if ssr < best_ssr:
                best_ssr = ssr
                best_threshold = t
        return {'threshold': best_threshold}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_viirs_data():
    """Load VIIRS nighttime light data"""
    print("  Loading VIIRS data...")
    try:
        df = pd.read_excel(PATHS['viirs'])
        print(f"    Columns: {list(df.columns)[:5]}...")
        viirs_dict = {}

        for city_en, city_info in CITIES.items():
            if city_info.get('is_municipality'):
                mask = df['省份名称'].str.contains(city_info['province'], na=False)
            else:
                mask = df['城市名称'].str.contains(city_info['cn_short'], na=False)

            city_data = df[mask]
            if len(city_data) > 0:
                yearly = city_data.groupby('年度')['DN均值'].mean()
                viirs_dict[city_en] = yearly.to_dict()
                print_status(f"{city_en}: {len(yearly)} years")

        return viirs_dict
    except Exception as e:
        print_status(f"VIIRS failed: {e}", False)
        return None


def load_co2_data():
    """Load CO2 emissions data"""
    print("  Loading CO2 data...")
    try:
        df = pd.read_excel(PATHS['co2'])
        print(f"    Columns: {list(df.columns)[:5]}...")
        co2_dict = {}

        for city_en, city_info in CITIES.items():
            mask = df['城市'].str.contains(city_info['cn_short'], na=False)
            city_data = df[mask]
            if len(city_data) > 0:
                yearly = {}
                for _, row in city_data.iterrows():
                    yearly[row['年份']] = row['CO2排放总量_吨'] / 1e8
                co2_dict[city_en] = yearly
                print_status(f"{city_en}: {len(yearly)} years")

        return co2_dict
    except Exception as e:
        print_status(f"CO2 failed: {e}", False)
        return None


def load_pm25_data():
    """Load PM2.5 data"""
    print("  Loading PM2.5 data...")
    try:
        df = pd.read_excel(PATHS['pm25'])
        print(f"    Columns: {list(df.columns)[:5]}...")
        pm25_dict = {}

        for city_en, city_info in CITIES.items():
            mask = df['CITY'].str.contains(city_info['cn_short'], na=False)
            city_data = df[mask]
            if len(city_data) > 0:
                yearly = {row['year']: row['pm2.5'] for _, row in city_data.iterrows()}
                pm25_dict[city_en] = yearly
                print_status(f"{city_en}: {len(yearly)} years")

        return pm25_dict
    except Exception as e:
        print_status(f"PM2.5 failed: {e}", False)
        return None


def load_population_data():
    """Load population density data"""
    print("  Loading population data...")
    try:
        df = pd.read_excel(PATHS['population'])
        pop_dict = {}

        for city_en, city_info in CITIES.items():
            mask = df['name'].str.contains(city_info['cn_short'], na=False)
            city_data = df[mask]
            if len(city_data) > 0:
                yearly = {row['时间']: row['数值'] for _, row in city_data.iterrows()}
                pop_dict[city_en] = yearly
                print_status(f"{city_en}: {len(yearly)} years")

        return pop_dict
    except Exception as e:
        print_status(f"Population failed: {e}", False)
        return None


def load_osm_parks():
    """Load OSM park data"""
    print("  Loading OSM park data...")
    try:
        import geopandas as gpd
    except ImportError:
        print_status("geopandas not installed", False)
        return None

    parks_dir = PATHS['osm_parks']
    if not parks_dir.exists():
        print_status(f"Directory not found: {parks_dir}", False)
        return None

    shp_files = sorted(parks_dir.glob("*.shp"))
    print(f"    Found {len(shp_files)} shapefiles")
    parks_dict = {city: {} for city in CITIES.keys()}

    for shp_file in shp_files:
        try:
            year = int(shp_file.stem.split('年')[0])
        except:
            for y in range(2014, 2024):
                if str(y) in shp_file.stem:
                    year = y
                    break
            else:
                continue

        if year < YEAR_START or year > YEAR_END:
            continue

        try:
            gdf = gpd.read_file(shp_file)
            if gdf.crs is None:
                gdf.set_crs('EPSG:4326', inplace=True)
            gdf_proj = gdf.to_crs('ESRI:102025')
            gdf['area_km2'] = gdf_proj.geometry.area / 1e6
            gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs != 'EPSG:4326' else gdf
            gdf['lon'] = gdf_wgs84.geometry.centroid.x
            gdf['lat'] = gdf_wgs84.geometry.centroid.y

            for city_en, city_info in CITIES.items():
                bbox = city_info['bbox']
                mask = ((gdf['lon'] >= bbox[0]) & (gdf['lon'] <= bbox[2]) &
                        (gdf['lat'] >= bbox[1]) & (gdf['lat'] <= bbox[3]))
                city_parks = gdf[mask]
                if len(city_parks) > 0:
                    parks_dict[city_en][year] = {
                        'area_km2': city_parks['area_km2'].sum(),
                        'count': len(city_parks)
                    }
            print_status(f"{year} processed")
        except Exception as e:
            print_status(f"{shp_file.name}: {e}", False)

    return parks_dict


def load_clcd_all():
    """Load CLCD land cover data"""
    print("  Loading CLCD data...")
    try:
        import rasterio
    except ImportError:
        print_status("rasterio not installed", False)
        return None

    clcd_dict = {city: {} for city in CITIES.keys()}

    for city_en, city_info in CITIES.items():
        city_cn = city_info['cn']
        for year in range(YEAR_START, YEAR_END + 1):
            file_path = PATHS['clcd'] / f"【立方数据学社】{city_cn}" / f"CLCD_v01_{year}_albert.tif"
            if not file_path.exists():
                continue
            try:
                with rasterio.open(file_path) as src:
                    data = src.read(1)
                    pixel_area_km2 = 0.0009
                    result = {
                        'Forest_km2': np.sum(data == 2) * pixel_area_km2,
                        'Grassland_km2': np.sum(data == 4) * pixel_area_km2,
                        'Water_km2': np.sum(data == 5) * pixel_area_km2,
                        'Wetland_km2': np.sum(data == 9) * pixel_area_km2,
                    }
                    result['Green_km2'] = result['Forest_km2'] + result['Grassland_km2']
                    result['Blue_km2'] = result['Water_km2'] + result['Wetland_km2']
                    result['GBI_km2'] = result['Green_km2'] + result['Blue_km2']
                    clcd_dict[city_en][year] = result
            except:
                continue

        if clcd_dict[city_en]:
            print_status(f"{city_en}: {len(clcd_dict[city_en])} years")

    return clcd_dict


def load_all_real_data():
    """Load all real data sources"""
    print_header(f"Loading REAL Data ({YEAR_START}-{YEAR_END})")

    viirs_dict = load_viirs_data()
    co2_dict = load_co2_data()
    pm25_dict = load_pm25_data()
    pop_dict = load_population_data()
    parks_dict = load_osm_parks()
    clcd_dict = load_clcd_all()

    all_records = []

    for city_en, city_info in CITIES.items():
        for year in range(YEAR_START, YEAR_END + 1):
            record = {'City': city_en, 'City_EN': city_info['en'], 'Year': year}

            if viirs_dict and city_en in viirs_dict:
                for y, v in viirs_dict[city_en].items():
                    if int(y) == year:
                        record['VIIRS'] = v
                        break

            if co2_dict and city_en in co2_dict and year in co2_dict[city_en]:
                record['CO2_100Mt'] = co2_dict[city_en][year]

            if pm25_dict and city_en in pm25_dict and year in pm25_dict[city_en]:
                record['PM25'] = pm25_dict[city_en][year]

            if pop_dict and city_en in pop_dict and year in pop_dict[city_en]:
                record['Pop_density'] = pop_dict[city_en][year]

            if parks_dict and city_en in parks_dict and year in parks_dict[city_en]:
                record['Park_km2'] = parks_dict[city_en][year]['area_km2']
                record['Park_count'] = parks_dict[city_en][year]['count']

            if clcd_dict and city_en in clcd_dict and year in clcd_dict[city_en]:
                for k, v in clcd_dict[city_en][year].items():
                    record[k] = v

            all_records.append(record)

    df = pd.DataFrame(all_records)

    print(f"\n  ★ Data loading complete: {len(df)} records")
    for col in ['VIIRS', 'CO2_100Mt', 'PM25', 'Park_km2', 'GBI_km2']:
        if col in df.columns:
            n = df[col].notna().sum()
            pct = n / len(df) * 100
            status = "✓" if pct > 50 else "⚠"
            print(f"    {status} {col}: {n}/{len(df)} ({pct:.0f}%)")

    return df


def generate_simulated_data():
    """Generate simulated data when real data unavailable"""
    print("\n  ⚠️ Generating SIMULATED data...")
    np.random.seed(42)
    records = []

    params = {
        'Beijing': {'viirs': 18, 'park': 80, 'pm25': 65, 'co2': 1.2, 'pop': 2150},
        'Shanghai': {'viirs': 22, 'park': 60, 'pm25': 45, 'co2': 1.5, 'pop': 2450},
        'Guangzhou': {'viirs': 16, 'park': 55, 'pm25': 40, 'co2': 0.9, 'pop': 1530},
        'Shenzhen': {'viirs': 25, 'park': 45, 'pm25': 35, 'co2': 0.7, 'pop': 1340},
        'Nanjing': {'viirs': 12, 'park': 50, 'pm25': 55, 'co2': 0.6, 'pop': 850},
        'Wuhan': {'viirs': 10, 'park': 45, 'pm25': 60, 'co2': 0.7, 'pop': 1120},
        'Chengdu': {'viirs': 8, 'park': 70, 'pm25': 50, 'co2': 0.5, 'pop': 1630},
        'Xian': {'viirs': 7, 'park': 35, 'pm25': 70, 'co2': 0.4, 'pop': 920},
        'Chongqing': {'viirs': 6, 'park': 40, 'pm25': 55, 'co2': 0.8, 'pop': 3200},
    }

    for year in range(YEAR_START, YEAR_END + 1):
        yf = (year - YEAR_START) / (YEAR_END - YEAR_START)
        for city, p in params.items():
            park_effect = p['park'] * (1 + 0.5 * yf) + np.random.normal(0, 5)
            viirs_value = p['viirs'] * (1 + 0.3 * yf) + np.random.normal(0, 1)
            pm25_base = p['pm25'] * (1 - 0.4 * yf)
            pm25_park_effect = -0.15 * np.log1p(park_effect)
            pm25_value = pm25_base + pm25_park_effect * 10 + np.random.normal(0, 3)

            record = {
                'City': city, 'City_EN': CITIES[city]['en'], 'Year': year,
                'VIIRS': viirs_value,
                'Park_km2': park_effect,
                'PM25': pm25_value,
                'CO2_100Mt': p['co2'] * (1 + 0.1 * yf) + 0.002 * park_effect + np.random.normal(0, 0.05),
                'Pop_density': p['pop'] + np.random.normal(0, 50),
                'Forest_km2': p['park'] * 0.8 * (1 + 0.3 * yf) + np.random.normal(0, 3),
                'Grassland_km2': p['park'] * 0.3 * (1 + 0.2 * yf) + np.random.normal(0, 2),
                'Water_km2': p['park'] * 0.2 + np.random.normal(0, 1),
                'Wetland_km2': p['park'] * 0.05 + np.random.normal(0, 0.5),
            }
            records.append(record)

    df = pd.DataFrame(records)
    df['Green_km2'] = df['Forest_km2'] + df['Grassland_km2']
    df['Blue_km2'] = df['Water_km2'] + df['Wetland_km2']
    df['GBI_km2'] = df['Green_km2'] + df['Blue_km2']
    df['Park_count'] = (df['Park_km2'] / 0.5).astype(int) + np.random.randint(10, 50, len(df))
    return df


def preprocess_data(df):
    """Compute derived variables"""
    print("  Computing derived variables...")

    if 'Park_count' in df.columns:
        df['Avg_Park_Size_km2'] = df['Park_km2'] / df['Park_count'].replace(0, np.nan)
        df['Park_Density'] = df['Park_count'] / (df['Park_km2'] + 1)
    else:
        df['Avg_Park_Size_km2'] = df['Park_km2'] / 100
        df['Park_Density'] = 100 / (df['Park_km2'] + 1)
        df['Park_count'] = 100

    if 'Blue_km2' in df.columns and 'Green_km2' in df.columns:
        df['Blue_Green_Ratio'] = df['Blue_km2'] / (df['Green_km2'] + 1)

    if 'Pop_density' in df.columns:
        df['GBI_Intensity'] = df.get('GBI_km2', df['Park_km2']) / (df['Pop_density'] + 1)
        df['Park_Per_Capita'] = df['Park_km2'] / (df['Pop_density'] + 1) * 1000

    df['VIIRS_Quartile'] = pd.qcut(df['VIIRS'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    viirs_median = df['VIIRS'].median()
    df['Dev_Regime'] = np.where(df['VIIRS'] > viirs_median, 'High_Dev', 'Low_Dev')
    df['Dev_Stage'] = np.where(df['VIIRS'] >= 9.0, 'Mature', 'Developing')

    # Add City_Code for mixed models
    city_codes = {city: i for i, city in enumerate(df['City'].unique())}
    df['City_Code'] = df['City'].map(city_codes)

    print(f"    VIIRS Median: {viirs_median:.2f}")
    return df, viirs_median


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def blue_green_comparison(df):
    results = []
    infra_types = [
        ('Park', 'Park_km2'), ('Forest', 'Forest_km2'), ('Grassland', 'Grassland_km2'),
        ('Water', 'Water_km2'), ('Blue_Total', 'Blue_km2'), ('Green_Total', 'Green_km2'),
        ('GBI_Total', 'GBI_km2'),
    ]

    for name, var in infra_types:
        if var not in df.columns:
            continue
        r_pm, p_pm, sig_pm = compute_correlations(df, var, 'PM25')
        r_co, p_co, sig_co = compute_correlations(df, var, 'CO2_100Mt')
        env_score = (-r_pm if not np.isnan(r_pm) else 0) * 0.6 + (-r_co if not np.isnan(r_co) else 0) * 0.4
        results.append({
            'Infrastructure': name, 'Variable': var,
            'PM25_r': r_pm, 'PM25_p': p_pm, 'PM25_sig': sig_pm,
            'CO2_r': r_co, 'CO2_p': p_co, 'CO2_sig': sig_co,
            'Env_Score': env_score,
        })
    return pd.DataFrame(results)


def run_gamm_analysis(df, gamm_analyzer):
    """Run GAMM (Generalized Additive Mixed Models) Analysis"""
    print_header("Running GAMM Analysis")
    results = {}

    # ==================== 1. Basic GAM: Park → PM2.5 ====================
    print("\n  [1] Fitting GAM: Park → PM2.5...")
    valid = df[['Park_km2', 'PM25']].dropna()
    if len(valid) > 20 and PYGAM_AVAILABLE:
        gam = gamm_analyzer.fit_gam_pygam(
            valid['Park_km2'].values,
            valid['PM25'].values,
            name='park_pm25',
            n_splines=20
        )
        if gam:
            results['gam_park_pm25'] = gamm_analyzer.results['gam_park_pm25']
            # Get partial dependence
            pdep = gamm_analyzer.get_partial_dependence('park_pm25', feature_idx=0)
            if pdep:
                results['pdep_park_pm25'] = pdep

    # ==================== 2. Basic GAM: Park → CO2 ====================
    print("\n  [2] Fitting GAM: Park → CO2...")
    valid = df[['Park_km2', 'CO2_100Mt']].dropna()
    if len(valid) > 20 and PYGAM_AVAILABLE:
        gam = gamm_analyzer.fit_gam_pygam(
            valid['Park_km2'].values,
            valid['CO2_100Mt'].values,
            name='park_co2',
            n_splines=20
        )
        if gam:
            results['gam_park_co2'] = gamm_analyzer.results['gam_park_co2']
            pdep = gamm_analyzer.get_partial_dependence('park_co2', feature_idx=0)
            if pdep:
                results['pdep_park_co2'] = pdep

    # ==================== 3. GAM with Interaction: Park × VIIRS → PM2.5 ====================
    print("\n  [3] Fitting GAM with Interaction: Park × VIIRS → PM2.5...")
    if PYGAM_AVAILABLE:
        gam_int = gamm_analyzer.fit_gam_with_interaction(
            df, 'Park_km2', 'PM25', 'VIIRS', name='park_viirs_pm25'
        )
        if gam_int:
            results['gam_interaction'] = gamm_analyzer.results.get('gam_interaction_park_viirs_pm25')

    # ==================== 4. GAMM with Random Effects (City) ====================
    print("\n  [4] Fitting GAMM with Random Effects (City)...")
    if STATSMODELS_AVAILABLE:
        # PM2.5 Model
        valid = df[['PM25', 'Park_km2', 'VIIRS', 'City']].dropna()
        if len(valid) > 30:
            gamm_pm25 = gamm_analyzer.fit_gamm_statsmodels(
                valid,
                formula="PM25 ~ Park_km2 + VIIRS",
                groups="City",
                name="pm25_mixed"
            )
            if gamm_pm25:
                results['gamm_pm25'] = gamm_analyzer.results['gamm_pm25_mixed']

        # CO2 Model
        valid = df[['CO2_100Mt', 'Park_km2', 'VIIRS', 'City']].dropna()
        if len(valid) > 30:
            gamm_co2 = gamm_analyzer.fit_gamm_statsmodels(
                valid,
                formula="CO2_100Mt ~ Park_km2 + VIIRS",
                groups="City",
                name="co2_mixed"
            )
            if gamm_co2:
                results['gamm_co2'] = gamm_analyzer.results['gamm_co2_mixed']

    # ==================== 5. GAM vs Linear Comparison ====================
    print("\n  [5] Comparing GAM vs Linear Model...")
    valid = df[['Park_km2', 'PM25']].dropna()
    if len(valid) > 20 and PYGAM_AVAILABLE:
        comparison = gamm_analyzer.compare_gam_vs_linear(
            valid['Park_km2'].values,
            valid['PM25'].values,
            name='park_pm25'
        )
        if comparison:
            results['gam_vs_linear'] = comparison
            print_status(f"GAM R² improvement: {comparison['improvement']['r2_gain']:.4f}")
            print_status(f"Nonlinearity significant: {comparison['improvement']['nonlinearity_significant']}")

    # ==================== 6. GAM Threshold Detection ====================
    print("\n  [6] Detecting Threshold using GAM...")
    valid = df[['Park_km2', 'PM25']].dropna()
    if len(valid) > 20 and PYGAM_AVAILABLE:
        threshold_result = gamm_analyzer.fit_threshold_gam(
            valid['Park_km2'].values,
            valid['PM25'].values,
            name='pm25'
        )
        if threshold_result:
            results['gam_threshold'] = threshold_result

    # ==================== 7. Multi-predictor GAM ====================
    print("\n  [7] Fitting Multi-predictor GAM...")
    multi_cols = ['Park_km2', 'VIIRS', 'Pop_density']
    available = [c for c in multi_cols if c in df.columns]
    valid = df[available + ['PM25']].dropna()
    if len(valid) > 30 and len(available) >= 2 and PYGAM_AVAILABLE:
        gam_multi = gamm_analyzer.fit_gam_pygam(
            valid[available].values,
            valid['PM25'].values,
            name='multivariate',
            n_splines=15
        )
        if gam_multi:
            results['gam_multivariate'] = gamm_analyzer.results['gam_multivariate']

    # ==================== 8. Stratified GAMM by Development Level ====================
    print("\n  [8] Fitting Stratified GAMM by Development Level...")
    results['stratified_gamm'] = {}
    for regime in ['High_Dev', 'Low_Dev']:
        regime_df = df[df['Dev_Regime'] == regime][['PM25', 'Park_km2', 'VIIRS', 'City']].dropna()
        if len(regime_df) > 15 and STATSMODELS_AVAILABLE:
            try:
                gamm_stratified = gamm_analyzer.fit_gamm_statsmodels(
                    regime_df,
                    formula="PM25 ~ Park_km2",
                    groups="City",
                    name=f"stratified_{regime}"
                )
                if gamm_stratified:
                    results['stratified_gamm'][regime] = gamm_analyzer.results[f'gamm_stratified_{regime}']
            except:
                pass

    return results


def run_deep_learning_analysis(df, dl_analyzer):
    """Run deep learning enhanced analysis"""
    print_header("Running Deep Learning Analysis")
    results = {}

    # 1. Neural Network for Park-PM25
    print("\n  Training Neural Network for Park-PM2.5...")
    valid = df[['Park_km2', 'PM25']].dropna()
    if len(valid) > 20:
        dl_analyzer.train_nonlinear_model(valid['Park_km2'].values, valid['PM25'].values, name='park_pm25', epochs=500)
        results['park_pm25_nn'] = True
        print_status("Park-PM25 neural network trained")

    # 2. Neural Network for Park-CO2
    print("\n  Training Neural Network for Park-CO2...")
    valid = df[['Park_km2', 'CO2_100Mt']].dropna()
    if len(valid) > 20:
        dl_analyzer.train_nonlinear_model(valid['Park_km2'].values, valid['CO2_100Mt'].values, name='park_co2',
                                          epochs=500)
        results['park_co2_nn'] = True
        print_status("Park-CO2 neural network trained")

    # 3. Deep Learning Threshold Detection
    print("\n  Detecting thresholds using deep learning...")
    valid = df[['Park_km2', 'PM25']].dropna()
    if len(valid) > 20:
        threshold_result = dl_analyzer.detect_threshold_dl(valid['Park_km2'].values, valid['PM25'].values,
                                                           name='pm25_threshold')
        results['dl_threshold_pm25'] = threshold_result
        print_status(f"PM25 threshold detected: {threshold_result['threshold']:.2f} km²")

    # 4. LSTM for temporal dynamics
    print("\n  Training LSTM for trajectory prediction...")
    feature_cols = ['VIIRS', 'Park_km2', 'Pop_density', 'CO2_100Mt']
    available_cols = [c for c in feature_cols if c in df.columns and df[c].notna().sum() > 20]
    if len(available_cols) >= 2:
        dl_analyzer.train_lstm_trajectories(df, 'PM25', available_cols, seq_length=3, epochs=200)
        results['lstm_trained'] = True
        print_status("LSTM trajectory model trained")

    # 5. Autoencoder for feature extraction
    print("\n  Training Autoencoder for feature extraction...")
    ae_features = ['Park_km2', 'VIIRS', 'PM25', 'CO2_100Mt', 'Pop_density']
    available_ae = [c for c in ae_features if c in df.columns and df[c].notna().sum() > 20]
    if len(available_ae) >= 3:
        dl_analyzer.train_autoencoder(df, available_ae, latent_dim=2, epochs=200)
        latent = dl_analyzer.get_latent_features(df, available_ae)
        if latent is not None:
            results['latent_features'] = latent
            print_status("Autoencoder feature extraction complete")

    # 6. VIIRS Stratified Analysis
    results['viirs_stratified'] = {}
    vq = df['VIIRS'].quantile([0.25, 0.5, 0.75])
    groups = [('Q1', df['VIIRS'] <= vq[0.25]), ('Q2', (df['VIIRS'] > vq[0.25]) & (df['VIIRS'] <= vq[0.5])),
              ('Q3', (df['VIIRS'] > vq[0.5]) & (df['VIIRS'] <= vq[0.75])), ('Q4', df['VIIRS'] > vq[0.75])]

    for name, mask in groups:
        g = df[mask][['Park_km2', 'PM25', 'CO2_100Mt', 'VIIRS']].dropna()
        if len(g) > 10:
            r_pm, p_pm = stats.pearsonr(g['Park_km2'], g['PM25'])
            r_co, p_co = stats.pearsonr(g['Park_km2'], g['CO2_100Mt'])
            results['viirs_stratified'][name] = {
                'n': len(g), 'viirs_range': (g['VIIRS'].min(), g['VIIRS'].max()), 'viirs_mean': g['VIIRS'].mean(),
                'park_pm25_r': r_pm, 'park_pm25_p': p_pm, 'park_co2_r': r_co, 'park_co2_p': p_co
            }

    # 7. Blue-Green Comparison
    results['blue_green'] = blue_green_comparison(df)

    # 8. Traditional Threshold Analysis
    if all(c in df.columns for c in ['VIIRS', 'Park_km2', 'CO2_100Mt']):
        vv = df['VIIRS'].dropna()
        if len(vv) > 30:
            vr = np.linspace(vv.quantile(0.15), vv.quantile(0.85), 30)
            tres = []
            for t in vr:
                lo = df[df['VIIRS'] <= t][['Park_km2', 'CO2_100Mt']].dropna()
                hi = df[df['VIIRS'] > t][['Park_km2', 'CO2_100Mt']].dropna()
                if len(lo) > 15 and len(hi) > 15:
                    r_lo, _ = stats.pearsonr(lo['Park_km2'], lo['CO2_100Mt'])
                    r_hi, _ = stats.pearsonr(hi['Park_km2'], hi['CO2_100Mt'])
                    tres.append({'viirs_threshold': t, 'r_below': r_lo, 'r_above': r_hi,
                                 'r_diff': r_hi - r_lo, 'n_below': len(lo), 'n_above': len(hi)})
            if tres:
                tdf = pd.DataFrame(tres)
                idx = tdf['r_diff'].abs().idxmax()
                results['threshold_analysis'] = {'optimal_threshold': tdf.loc[idx, 'viirs_threshold'],
                                                 'threshold_curve': tdf}

    return results


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================
def create_fig1_gamm_analysis(df, gamm_results, dl_results, viirs_median, gamm_analyzer, dl_analyzer, output_path):
    """Figure 1: GAMM Analysis - True GAMM with Deep Learning Comparison"""
    print("\nGenerating Fig1 (True GAMM Analysis)...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    vm = viirs_median

    # (a) GAM Smooth: Park → PM2.5 with Confidence Intervals
    ax = axes[0, 0]
    if all(c in df.columns for c in ['Park_km2', 'PM25', 'VIIRS']):
        v = df[['Park_km2', 'PM25', 'VIIRS']].dropna()
        if len(v) > 5:
            # Scatter with VIIRS coloring
            sc = ax.scatter(v['Park_km2'], v['PM25'], c=v['VIIRS'], cmap='RdYlBu_r', s=70, alpha=0.6,
                            edgecolors='white', linewidth=0.5, zorder=2)
            plt.colorbar(sc, ax=ax, shrink=0.8).set_label('VIIRS')

            # GAM smooth curve with CI
            if 'pdep_park_pm25' in gamm_results:
                pdep = gamm_results['pdep_park_pm25']
                ax.fill_between(pdep['x'], pdep['ci_lower'], pdep['ci_upper'],
                                color=COLORS['gamm'], alpha=0.2, label='95% CI', zorder=3)
                ax.plot(pdep['x'], pdep['y'], color=COLORS['gamm'], lw=3,
                        label='GAM Smooth', zorder=4)

            # Deep Learning comparison
            if 'park_pm25' in dl_analyzer.models:
                x_smooth = np.linspace(v['Park_km2'].min(), v['Park_km2'].max(), 100)
                y_nn = dl_analyzer.predict_nonlinear(x_smooth, 'park_pm25')
                if y_nn is not None:
                    ax.plot(x_smooth, y_nn, color=COLORS['neural'], lw=2, ls='--',
                            label='Neural Network', zorder=5, alpha=0.8)

            # Statistics
            if 'gam_park_pm25' in gamm_results:
                r2 = gamm_results['gam_park_pm25']['pseudo_r2']
                edof = gamm_results['gam_park_pm25']['edof']
                ax.text(0.03, 0.97, f'GAM R² = {r2:.3f}\nedf = {edof:.1f}',
                        transform=ax.transAxes, fontsize=10, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            ax.legend(loc='upper right', fontsize=9)

    ax.set_xlabel('Park Area (km²)', fontsize=11)
    ax.set_ylabel('PM$_{2.5}$ (μg/m³)', fontsize=11)
    ax.set_title('(a) GAM: Park Area → PM$_{2.5}$ with 95% CI', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (b) GAM vs Linear Model Comparison
    ax = axes[0, 1]
    if 'gam_vs_linear' in gamm_results:
        comp = gamm_results['gam_vs_linear']
        models = ['Linear', 'GAM']
        r2_values = [comp['linear']['r2'], comp['gam']['r2']]
        aic_values = [comp['linear']['aic'], comp['gam']['aic']]

        x_pos = np.arange(len(models))
        width = 0.35

        # R² bars
        bars1 = ax.bar(x_pos - width / 2, r2_values, width, label='R²', color=COLORS['blue'], alpha=0.8)

        # Add value labels
        for bar, val in zip(bars1, r2_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        # Secondary axis for AIC
        ax2 = ax.twinx()
        bars2 = ax2.bar(x_pos + width / 2, aic_values, width, label='AIC', color=COLORS['green'], alpha=0.8)

        for bar, val in zip(bars2, aic_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=10)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.set_ylabel('R²', fontsize=11, color=COLORS['blue'])
        ax2.set_ylabel('AIC', fontsize=11, color=COLORS['green'])

        # Nonlinearity significance
        is_sig = comp['improvement']['nonlinearity_significant']
        sig_text = "✓ Significant" if is_sig else "✗ Not Significant"
        ax.text(0.5, 0.95, f'Nonlinearity: {sig_text}\nΔR² = {comp["improvement"]["r2_gain"]:.4f}',
                transform=ax.transAxes, fontsize=10, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)

    ax.set_title('(b) GAM vs Linear Model Comparison', fontsize=12, fontweight='bold')

    # (c) GAMM Random Effects by City
    ax = axes[1, 0]
    if 'gamm_pm25' in gamm_results:
        re = gamm_results['gamm_pm25'].get('random_effects', {})
        if re:
            cities = list(re.keys())
            # Extract intercept random effects
            re_values = []
            for city in cities:
                city_re = re[city]
                if isinstance(city_re, dict):
                    re_values.append(list(city_re.values())[0] if city_re else 0)
                else:
                    re_values.append(0)

            # Sort by random effect
            sorted_idx = np.argsort(re_values)
            cities_sorted = [cities[i] for i in sorted_idx]
            re_sorted = [re_values[i] for i in sorted_idx]

            colors = [COLORS['High_Dev'] if df[df['City'] == c]['VIIRS'].mean() > vm else COLORS['Low_Dev']
                      for c in cities_sorted]

            ax.barh(range(len(cities_sorted)), re_sorted, color=colors, alpha=0.8)
            ax.set_yticks(range(len(cities_sorted)))
            ax.set_yticklabels(cities_sorted)
            ax.axvline(x=0, color='black', lw=1)

            # Legend
            ax.legend(handles=[Patch(facecolor=COLORS['High_Dev'], label='High Dev'),
                               Patch(facecolor=COLORS['Low_Dev'], label='Low Dev')],
                      loc='lower right', fontsize=9)

            # Model info
            if 'gamm_pm25' in gamm_results:
                aic = gamm_results['gamm_pm25']['aic']
                ax.text(0.02, 0.98, f'GAMM AIC = {aic:.1f}',
                        transform=ax.transAxes, fontsize=10, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Random Effect (City Intercept)', fontsize=11)
    ax.set_title('(c) GAMM: City-Level Random Effects', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # (d) GAM Threshold Detection
    ax = axes[1, 1]
    if 'gam_threshold' in gamm_results:
        thresh = gamm_results['gam_threshold']
        x_grid = thresh['x_grid']
        y_pred = thresh['y_pred']
        threshold_val = thresh['threshold']

        # Scatter original data
        v = df[['Park_km2', 'PM25']].dropna()
        ax.scatter(v['Park_km2'], v['PM25'], c='gray', alpha=0.4, s=40, label='Data', zorder=1)

        # GAM curve
        ax.plot(x_grid, y_pred, color=COLORS['gamm'], lw=3, label='GAM Smooth', zorder=2)

        # Threshold line
        ax.axvline(x=threshold_val, color=COLORS['threshold'], ls='--', lw=2.5,
                   label=f'GAM Threshold: {threshold_val:.0f} km²', zorder=3)

        # DL threshold comparison
        if 'dl_threshold_pm25' in dl_results:
            dl_thresh = dl_results['dl_threshold_pm25']['threshold']
            ax.axvline(x=dl_thresh, color=COLORS['neural'], ls=':', lw=2,
                       label=f'DL Threshold: {dl_thresh:.0f} km²', zorder=3)

        ax.legend(loc='upper right', fontsize=9)

        # R² annotation
        r2 = thresh['pseudo_r2']
        ax.text(0.02, 0.98, f'GAM R² = {r2:.3f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Park Area (km²)', fontsize=11)
    ax.set_ylabel('PM$_{2.5}$ (μg/m³)', fontsize=11)
    ax.set_title('(d) GAM-Based Threshold Detection', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 1: GAMM Analysis - Nonlinear Relationships with Mixed Effects',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print_status(f"Figure 1 saved: {output_path}")


def create_fig2_blue_green_analysis(df, dl_results, gamm_analyzer, output_path):
    """Figure 2: Blue-Green Infrastructure Comparison"""
    print("\nGenerating Fig2 (Blue-Green Analysis)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    bg_results = dl_results.get('blue_green', pd.DataFrame())

    # (a) PM2.5 Correlation by GBI Type
    ax = axes[0, 0]
    infra_order = ['Water', 'Blue_Total', 'Forest', 'Park', 'Grassland', 'Green_Total']
    if len(bg_results) > 0:
        infra_data = bg_results[bg_results['Infrastructure'].isin(infra_order)].copy()
        if len(infra_data) > 0:
            infra_data['Order'] = infra_data['Infrastructure'].map({v: i for i, v in enumerate(infra_order)})
            infra_data = infra_data.sort_values('Order')
            colors = [COLORS['blue'] if inf in ['Water', 'Blue_Total'] else COLORS['green']
                      for inf in infra_data['Infrastructure']]
            ax.barh(range(len(infra_data)), infra_data['PM25_r'], color=colors, alpha=0.8)
            ax.set_yticks(range(len(infra_data)))
            ax.set_yticklabels(infra_data['Infrastructure'])
            ax.axvline(x=0, color='black', linewidth=0.5)
            for i, (_, row) in enumerate(infra_data.iterrows()):
                x_pos = row['PM25_r'] + 0.02 if row['PM25_r'] > 0 else row['PM25_r'] - 0.05
                ax.text(x_pos, i, row['PM25_sig'], va='center', fontsize=10, fontweight='bold')
            ax.legend(handles=[Patch(facecolor=COLORS['blue'], label='Blue Infrastructure'),
                               Patch(facecolor=COLORS['green'], label='Green Infrastructure')],
                      loc='lower right', fontsize=9)
    ax.set_xlabel('Correlation with PM$_{2.5}$ (r)', fontsize=11)
    ax.set_title('(a) GBI Type → PM$_{2.5}$ Association', fontsize=12, fontweight='bold')

    # (b) CO2 Correlation by GBI Type
    ax = axes[0, 1]
    if len(bg_results) > 0 and 'infra_data' in dir():
        colors = [COLORS['blue'] if inf in ['Water', 'Blue_Total'] else COLORS['green']
                  for inf in infra_data['Infrastructure']]
        ax.barh(range(len(infra_data)), infra_data['CO2_r'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(infra_data)))
        ax.set_yticklabels(infra_data['Infrastructure'])
        ax.axvline(x=0, color='black', linewidth=0.5)
        for i, (_, row) in enumerate(infra_data.iterrows()):
            x_pos = row['CO2_r'] + 0.02 if row['CO2_r'] > 0 else row['CO2_r'] - 0.05
            ax.text(x_pos, i, row['CO2_sig'], va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Correlation with CO$_2$ (r)', fontsize=11)
    ax.set_title('(b) GBI Type → CO$_2$ Association', fontsize=12, fontweight='bold')

    # (c) Environmental Performance Score (GAM-weighted)
    ax = axes[1, 0]
    if len(bg_results) > 0 and 'infra_data' in dir():
        score_data = infra_data.sort_values('Env_Score', ascending=True)
        colors_sorted = [COLORS['blue'] if inf in ['Water', 'Blue_Total'] else COLORS['green']
                         for inf in score_data['Infrastructure']]
        ax.barh(range(len(score_data)), score_data['Env_Score'], color=colors_sorted, alpha=0.8)
        ax.set_yticks(range(len(score_data)))
        ax.set_yticklabels(score_data['Infrastructure'])
        ax.axvline(x=0, color='black', linewidth=0.5)

        # Calculate blue/green ratio
        blue_score = score_data[score_data['Infrastructure'].isin(['Water', 'Blue_Total'])]['Env_Score'].mean()
        green_score = score_data[~score_data['Infrastructure'].isin(['Water', 'Blue_Total'])]['Env_Score'].mean()
        if green_score != 0 and not np.isnan(green_score):
            ratio = abs(blue_score / green_score)
            ax.text(0.95, 0.05, f'Blue/Green Ratio: {ratio:.2f}x',
                    transform=ax.transAxes, fontsize=10, ha='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.set_xlabel('Environmental Performance Score', fontsize=11)
    ax.set_title('(c) Integrated Environmental Performance', fontsize=12, fontweight='bold')

    # (d) GAM Partial Dependence: Blue vs Green
    ax = axes[1, 1]
    # Fit GAMs for blue and green and compare
    if 'Blue_km2' in df.columns and 'Green_km2' in df.columns and PYGAM_AVAILABLE:
        # Blue GAM
        valid_blue = df[['Blue_km2', 'PM25']].dropna()
        if len(valid_blue) > 15:
            gamm_analyzer.fit_gam_pygam(valid_blue['Blue_km2'].values, valid_blue['PM25'].values,
                                        name='blue_pm25', n_splines=15)
            pdep_blue = gamm_analyzer.get_partial_dependence('blue_pm25', feature_idx=0)
            if pdep_blue:
                ax.plot(pdep_blue['x'], pdep_blue['y'], color=COLORS['blue'], lw=2.5, label='Blue Infrastructure')
                ax.fill_between(pdep_blue['x'], pdep_blue['ci_lower'], pdep_blue['ci_upper'],
                                color=COLORS['blue'], alpha=0.2)

        # Green GAM
        valid_green = df[['Green_km2', 'PM25']].dropna()
        if len(valid_green) > 15:
            gamm_analyzer.fit_gam_pygam(valid_green['Green_km2'].values, valid_green['PM25'].values,
                                        name='green_pm25', n_splines=15)
            pdep_green = gamm_analyzer.get_partial_dependence('green_pm25', feature_idx=0)
            if pdep_green:
                ax.plot(pdep_green['x'], pdep_green['y'], color=COLORS['green'], lw=2.5, label='Green Infrastructure')
                ax.fill_between(pdep_green['x'], pdep_green['ci_lower'], pdep_green['ci_upper'],
                                color=COLORS['green'], alpha=0.2)

        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlabel('Infrastructure Area (km²)', fontsize=11)
        ax.set_ylabel('Partial Effect on PM$_{2.5}$', fontsize=11)
    ax.set_title('(d) GAM Partial Dependence: Blue vs Green', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 2: Blue-Green Infrastructure Performance Comparison',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print_status(f"Figure 2 saved: {output_path}")


def create_fig3_threshold_analysis(df, gamm_results, dl_results, gamm_analyzer, dl_analyzer, output_path):
    """Figure 3: Threshold and Regime Analysis"""
    print("\nGenerating Fig3 (Threshold Analysis)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) GAM Derivative Analysis for Threshold
    ax = axes[0, 0]
    if 'gam_threshold' in gamm_results:
        thresh = gamm_results['gam_threshold']
        x_grid = thresh['x_grid'][:-2]  # Align with second derivative
        second_deriv = thresh['second_derivative']

        # Plot second derivative (curvature)
        ax.plot(x_grid, second_deriv, color=COLORS['gamm'], lw=2, label='Second Derivative (Curvature)')
        ax.fill_between(x_grid, second_deriv, 0, alpha=0.3, color=COLORS['gamm'])

        # Mark threshold
        threshold_idx = np.argmax(np.abs(second_deriv))
        ax.axvline(x=x_grid[threshold_idx], color=COLORS['threshold'], ls='--', lw=2.5,
                   label=f'Max Curvature: {x_grid[threshold_idx]:.0f} km²')
        ax.scatter([x_grid[threshold_idx]], [second_deriv[threshold_idx]],
                   c=COLORS['threshold'], s=100, zorder=5, edgecolors='black')

        ax.axhline(y=0, color='gray', ls='-', lw=0.5)
        ax.legend(loc='upper right', fontsize=9)

    ax.set_xlabel('Park Area (km²)', fontsize=11)
    ax.set_ylabel('Second Derivative (d²y/dx²)', fontsize=11)
    ax.set_title('(a) GAM Curvature Analysis for Threshold', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (b) Stratified GAMM Coefficients
    ax = axes[0, 1]
    if 'stratified_gamm' in gamm_results:
        strat = gamm_results['stratified_gamm']
        regimes = list(strat.keys())
        if regimes:
            coeffs = []
            for regime in regimes:
                fe = strat[regime].get('fe_params', {})
                park_coef = fe.get('Park_km2', 0)
                coeffs.append(park_coef)

            colors = [COLORS.get(r, '#888') for r in regimes]
            bars = ax.bar(regimes, coeffs, color=colors, alpha=0.8)

            for bar, coef in zip(bars, coeffs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f'{coef:.4f}', ha='center', va='bottom', fontsize=10)

            ax.axhline(y=0, color='black', lw=0.5)
            ax.set_ylabel('Park → PM$_{2.5}$ Coefficient', fontsize=11)
            ax.set_title('(b) GAMM Coefficients by Development Level', fontsize=12, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')

    # (c) PM2.5 Trends by Regime
    ax = axes[1, 0]
    if 'Dev_Regime' in df.columns and 'PM25' in df.columns:
        yr = df.groupby(['Year', 'Dev_Regime'])['PM25'].mean().reset_index()
        for regime in ['High_Dev', 'Low_Dev']:
            rd = yr[yr['Dev_Regime'] == regime]
            if len(rd) > 0:
                ax.plot(rd['Year'], rd['PM25'], marker='o', lw=2, ms=7, color=COLORS[regime],
                        label=regime.replace('_', ' '))

        # Add trend lines with GAM
        if PYGAM_AVAILABLE:
            for regime in ['High_Dev', 'Low_Dev']:
                rd = yr[yr['Dev_Regime'] == regime]
                if len(rd) > 3:
                    from pygam import LinearGAM, s
                    gam = LinearGAM(s(0, n_splines=5)).fit(rd['Year'].values.reshape(-1, 1), rd['PM25'].values)
                    x_smooth = np.linspace(rd['Year'].min(), rd['Year'].max(), 50)
                    y_smooth = gam.predict(x_smooth.reshape(-1, 1))
                    ax.plot(x_smooth, y_smooth, ls='--', lw=1.5, color=COLORS[regime], alpha=0.7)

        ax.legend(loc='upper right', fontsize=10)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('PM$_{2.5}$ (μg/m³)', fontsize=11)
    ax.set_title('(c) PM$_{2.5}$ Temporal Trends by Regime (GAM Smoothed)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (d) Phase Transition Analysis
    ax = axes[1, 1]
    if 'threshold_analysis' in dl_results:
        td = dl_results['threshold_analysis']['threshold_curve']
        opt = dl_results['threshold_analysis']['optimal_threshold']

        ax.plot(td['viirs_threshold'], td['r_below'], color=COLORS['Developing'], lw=2.5, marker='o', ms=4,
                label='Below VIIRS Threshold')
        ax.plot(td['viirs_threshold'], td['r_above'], color=COLORS['Mature'], lw=2.5, marker='s', ms=4,
                label='Above VIIRS Threshold')
        ax.plot(td['viirs_threshold'], td['r_diff'], color=COLORS['threshold'], lw=2, ls=':', marker='^', ms=3,
                label='Δr (Difference)', alpha=0.8)

        ax.axvline(x=opt, color='green', ls='--', lw=2, label=f'Optimal VIIRS: {opt:.1f}')
        ax.axhline(y=0, color='gray', ls='-', lw=1, alpha=0.5)

        # Add GAM threshold if available
        if 'gam_threshold' in gamm_results:
            gam_thresh = gamm_results['gam_threshold']['threshold']
            ax.axvline(x=gam_thresh, color=COLORS['gamm'], ls=':', lw=2,
                       label=f'GAM Park Threshold: {gam_thresh:.0f}')

        ax.legend(fontsize=8, loc='upper left')

    ax.set_xlabel('VIIRS Threshold', fontsize=11)
    ax.set_ylabel('Park-CO$_2$ Correlation (r)', fontsize=11)
    ax.set_title('(d) Development Phase Transition Analysis', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.8, 1.0)

    fig.suptitle('Figure 3: Threshold-Based Regime Transition Analysis',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print_status(f"Figure 3 saved: {output_path}")


def create_fig4_city_analysis(df, viirs_median, gamm_analyzer, dl_analyzer, output_path):
    """Figure 4: City-Level Analysis"""
    print("\nGenerating Fig4 (City Analysis)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    vm = viirs_median

    # (a) City VIIRS Ranking
    ax = axes[0, 0]
    if 'VIIRS' in df.columns:
        cv = df.groupby('City')['VIIRS'].mean().sort_values(ascending=True)
        colors = [COLORS['High_Dev'] if v > vm else COLORS['Low_Dev'] for v in cv.values]
        ax.barh(range(len(cv)), cv.values, color=colors, height=0.7)
        ax.set_yticks(range(len(cv)))
        ax.set_yticklabels(cv.index)
        ax.axvline(x=vm, color=COLORS['threshold'], ls='--', lw=2, label=f'Median: {vm:.1f}')
        for i, (city, v) in enumerate(cv.items()):
            ax.text(v + 0.3, i, f'{v:.1f}', va='center', fontsize=9)
        ax.legend(handles=[Patch(facecolor=COLORS['High_Dev'], label='High Dev'),
                           Patch(facecolor=COLORS['Low_Dev'], label='Low Dev'),
                           Line2D([0], [0], color=COLORS['threshold'], linestyle='--', label=f'Median: {vm:.1f}')],
                  loc='lower right', fontsize=9)
    ax.set_xlabel('Average VIIRS', fontsize=11)
    ax.set_title('(a) City Development Classification', fontsize=12, fontweight='bold')

    # (b) Park Area by City (Latest Year)
    ax = axes[0, 1]
    if 'Park_km2' in df.columns:
        latest = df[df['Year'] == YEAR_END][['City', 'Park_km2', 'Dev_Regime']].dropna()
        if len(latest) > 0:
            latest = latest.sort_values('Park_km2', ascending=True)
            colors = [COLORS.get(r, '#888') for r in latest['Dev_Regime']]
            ax.barh(range(len(latest)), latest['Park_km2'].values, color=colors, height=0.7)
            ax.set_yticks(range(len(latest)))
            ax.set_yticklabels(latest['City'].values)
            ax.legend(handles=[Patch(facecolor=COLORS['High_Dev'], label='High Dev'),
                               Patch(facecolor=COLORS['Low_Dev'], label='Low Dev')],
                      loc='lower right', fontsize=9)
    ax.set_xlabel(f'Park Area {YEAR_END} (km²)', fontsize=11)
    ax.set_title('(b) Park Area by City', fontsize=12, fontweight='bold')

    # (c) City Trajectories with GAM Smoothing
    ax = axes[1, 0]
    if 'VIIRS' in df.columns:
        ax.axhline(y=vm, color='gray', ls='--', lw=1.5, alpha=0.7)

        for city in df['City'].unique():
            cd = df[df['City'] == city].sort_values('Year')
            if len(cd) > 0 and 'VIIRS' in cd.columns:
                avg = cd['VIIRS'].mean()
                color = COLORS['High_Dev'] if avg >= vm else COLORS['Low_Dev']

                # Original points
                ax.scatter(cd['Year'], cd['VIIRS'], c=color, s=30, alpha=0.6)

                # GAM smooth for each city
                if len(cd) > 3 and PYGAM_AVAILABLE:
                    try:
                        from pygam import LinearGAM, s
                        gam = LinearGAM(s(0, n_splines=4)).fit(
                            cd['Year'].values.reshape(-1, 1),
                            cd['VIIRS'].values
                        )
                        x_smooth = np.linspace(cd['Year'].min(), cd['Year'].max(), 30)
                        y_smooth = gam.predict(x_smooth.reshape(-1, 1))
                        ax.plot(x_smooth, y_smooth, color=color, lw=1.5, alpha=0.8)
                    except:
                        ax.plot(cd['Year'], cd['VIIRS'], color=color, lw=1, alpha=0.6)
                else:
                    ax.plot(cd['Year'], cd['VIIRS'], color=color, lw=1, alpha=0.6)

                ax.annotate(city, (cd['Year'].iloc[-1], cd['VIIRS'].iloc[-1]),
                            xytext=(5, 0), textcoords='offset points', fontsize=8, color=color)

        ax.set_xlim(YEAR_START - 0.5, YEAR_END + 1.5)
        legend_elements = [Patch(facecolor=COLORS['High_Dev'], label='High Dev'),
                           Patch(facecolor=COLORS['Low_Dev'], label='Low Dev'),
                           Line2D([0], [0], color='gray', ls='--', label=f'Median: {vm:.1f}')]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('VIIRS', fontsize=11)
    ax.set_title('(c) City VIIRS Trajectories (GAM Smoothed)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (d) Park Growth vs PM2.5 Reduction
    ax = axes[1, 1]
    if 'Park_km2' in df.columns and 'PM25' in df.columns:
        summary = {}
        for city in df['City'].unique():
            cd = df[df['City'] == city].sort_values('Year')
            pk = cd['Park_km2'].dropna()
            pm = cd['PM25'].dropna()
            if len(pk) >= 2 and len(pm) >= 2 and pk.iloc[0] > 0 and pm.iloc[0] > 0:
                summary[city] = {
                    'park_growth': (pk.iloc[-1] - pk.iloc[0]) / pk.iloc[0] * 100,
                    'pm_change': (pm.iloc[-1] - pm.iloc[0]) / pm.iloc[0] * 100,
                    'regime': cd['Dev_Regime'].iloc[0] if 'Dev_Regime' in cd.columns else 'Unknown'
                }

        if summary:
            for city, d in summary.items():
                color = COLORS.get(d['regime'], '#888')
                ax.scatter(d['park_growth'], d['pm_change'], c=color, s=100, alpha=0.8, edgecolors='black')
                ax.annotate(city, (d['park_growth'], d['pm_change']),
                            xytext=(5, 5), textcoords='offset points', fontsize=9)

            # GAM trend line
            x_vals = np.array([d['park_growth'] for d in summary.values()])
            y_vals = np.array([d['pm_change'] for d in summary.values()])

            if len(x_vals) > 5 and PYGAM_AVAILABLE:
                from pygam import LinearGAM, s
                gam = LinearGAM(s(0, n_splines=5)).fit(x_vals.reshape(-1, 1), y_vals)
                x_smooth = np.linspace(x_vals.min(), x_vals.max(), 50)
                y_smooth = gam.predict(x_smooth.reshape(-1, 1))
                ci = gam.confidence_intervals(x_smooth.reshape(-1, 1), width=0.95)
                ax.fill_between(x_smooth, ci[:, 0], ci[:, 1], color=COLORS['gamm'], alpha=0.2)
                ax.plot(x_smooth, y_smooth, color=COLORS['gamm'], lw=2.5, label='GAM Trend')
            elif len(x_vals) > 3:
                z = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), 'k--', lw=1.5, alpha=0.7, label='Linear Trend')

        ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
        ax.legend(handles=[Patch(facecolor=COLORS['High_Dev'], label='High Dev'),
                           Patch(facecolor=COLORS['Low_Dev'], label='Low Dev'),
                           Line2D([0], [0], color=COLORS['gamm'], lw=2, label='GAM Trend')],
                  loc='upper right', fontsize=9)

    ax.set_xlabel('Park Growth (%)', fontsize=11)
    ax.set_ylabel('PM$_{2.5}$ Change (%)', fontsize=11)
    ax.set_title('(d) Park Expansion vs PM$_{2.5}$ Reduction', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 4: City-Level Development and GBI Performance',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print_status(f"Figure 4 saved: {output_path}")


def create_fig5_method_comparison(df, gamm_results, dl_results, gamm_analyzer, dl_analyzer, output_path):
    """Figure 5: Method Comparison - GAMM vs Deep Learning"""
    print("\nGenerating Fig5 (Method Comparison)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    valid = df[['Park_km2', 'PM25']].dropna()
    x_range = np.linspace(valid['Park_km2'].min(), valid['Park_km2'].max(), 100)

    # (a) All Methods Comparison
    ax = axes[0, 0]
    ax.scatter(valid['Park_km2'], valid['PM25'], c='gray', alpha=0.4, s=40, label='Data')

    # Linear
    z = np.polyfit(valid['Park_km2'], valid['PM25'], 1)
    ax.plot(x_range, np.polyval(z, x_range), 'k--', lw=2, label='Linear', alpha=0.7)

    # Polynomial
    z_poly = np.polyfit(valid['Park_km2'], valid['PM25'], 3)
    ax.plot(x_range, np.polyval(z_poly, x_range), color='orange', lw=2, ls='-.', label='Polynomial (deg=3)')

    # GAM
    if 'pdep_park_pm25' in gamm_results:
        pdep = gamm_results['pdep_park_pm25']
        ax.plot(pdep['x'], pdep['y'], color=COLORS['gamm'], lw=3, label='GAM')

    # Neural Network
    if 'park_pm25' in dl_analyzer.models:
        y_nn = dl_analyzer.predict_nonlinear(x_range, 'park_pm25')
        if y_nn is not None:
            ax.plot(x_range, y_nn, color=COLORS['neural'], lw=2.5, ls=':', label='Neural Network')

    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('Park Area (km²)', fontsize=11)
    ax.set_ylabel('PM$_{2.5}$ (μg/m³)', fontsize=11)
    ax.set_title('(a) Method Comparison: Park → PM$_{2.5}$', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (b) R² Comparison
    ax = axes[0, 1]
    methods = ['Linear', 'GAM', 'Neural Net']
    r2_values = []

    # Linear R²
    y_lin = np.polyval(z, valid['Park_km2'])
    ss_res_lin = np.sum((valid['PM25'] - y_lin) ** 2)
    ss_tot = np.sum((valid['PM25'] - valid['PM25'].mean()) ** 2)
    r2_lin = 1 - ss_res_lin / ss_tot
    r2_values.append(r2_lin)

    # GAM R²
    r2_gam = gamm_results.get('gam_park_pm25', {}).get('pseudo_r2', 0)
    r2_values.append(r2_gam)

    # Neural Net R² (approximated)
    if 'park_pm25' in dl_analyzer.models:
        y_nn = dl_analyzer.predict_nonlinear(valid['Park_km2'].values, 'park_pm25')
        if y_nn is not None:
            ss_res_nn = np.sum((valid['PM25'].values - y_nn) ** 2)
            r2_nn = 1 - ss_res_nn / ss_tot
            r2_values.append(r2_nn)
        else:
            r2_values.append(0)
    else:
        r2_values.append(0)

    colors = ['gray', COLORS['gamm'], COLORS['neural']]
    bars = ax.bar(methods, r2_values, color=colors, alpha=0.8)
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('R² (Explained Variance)', fontsize=11)
    ax.set_title('(b) Model Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(r2_values) * 1.2)

    # (c) Threshold Comparison
    ax = axes[1, 0]
    threshold_methods = []
    threshold_values = []

    # GAM threshold
    if 'gam_threshold' in gamm_results:
        threshold_methods.append('GAM\n(Curvature)')
        threshold_values.append(gamm_results['gam_threshold']['threshold'])

    # DL threshold
    if 'dl_threshold_pm25' in dl_results:
        threshold_methods.append('Neural Net\n(Gradient)')
        threshold_values.append(dl_results['dl_threshold_pm25']['threshold'])

    # Traditional threshold
    if 'threshold_analysis' in dl_results:
        threshold_methods.append('Piecewise\n(Traditional)')
        # Estimate traditional threshold
        traditional_thresh = dl_results['threshold_analysis']['optimal_threshold']
        # Convert VIIRS to park equivalent (rough approximation)
        viirs_park_ratio = df['Park_km2'].mean() / df['VIIRS'].mean()
        threshold_values.append(traditional_thresh * viirs_park_ratio)

    if threshold_methods:
        colors = [COLORS['gamm'], COLORS['neural'], COLORS['threshold']][:len(threshold_methods)]
        bars = ax.bar(threshold_methods, threshold_values, color=colors, alpha=0.8)
        for bar, val in zip(bars, threshold_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Park Threshold (km²)', fontsize=11)
        ax.set_title('(c) Threshold Detection Comparison', fontsize=12, fontweight='bold')

    # (d) Confidence/Uncertainty Comparison
    ax = axes[1, 1]

    # GAM with CI
    if 'pdep_park_pm25' in gamm_results:
        pdep = gamm_results['pdep_park_pm25']
        ax.fill_between(pdep['x'], pdep['ci_lower'], pdep['ci_upper'],
                        color=COLORS['gamm'], alpha=0.3, label='GAM 95% CI')
        ax.plot(pdep['x'], pdep['y'], color=COLORS['gamm'], lw=2, label='GAM')

    # Neural Network (no native CI, use bootstrap approximation note)
    if 'park_pm25' in dl_analyzer.models:
        y_nn = dl_analyzer.predict_nonlinear(x_range, 'park_pm25')
        if y_nn is not None:
            ax.plot(x_range, y_nn, color=COLORS['neural'], lw=2, ls='--', label='Neural Net (point estimate)')

    ax.scatter(valid['Park_km2'], valid['PM25'], c='gray', alpha=0.3, s=30, zorder=1)

    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('Park Area (km²)', fontsize=11)
    ax.set_ylabel('PM$_{2.5}$ (μg/m³)', fontsize=11)
    ax.set_title('(d) Uncertainty Quantification: GAM vs Neural Net', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add annotation about GAM advantages
    ax.text(0.02, 0.02, 'GAM provides built-in\nconfidence intervals',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.suptitle('Figure 5: GAMM vs Deep Learning Method Comparison',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print_status(f"Figure 5 saved: {output_path}")


def export_results_to_excel(df, gamm_results, dl_results, gamm_analyzer, dl_analyzer, output_path):
    """Export all results to Excel"""
    print("\nExporting results to Excel...")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Raw Data
        df.to_excel(writer, sheet_name='Raw_Data', index=False)

        # Summary Statistics
        summary_data = {
            'Metric': [
                'Total Observations', 'Cities', 'Year Range', 'VIIRS Median',
                'Park-PM25 Correlation', 'GAMM Available', 'Deep Learning Available',
                'GAM Models Trained', 'DL Models Trained'
            ],
            'Value': [
                len(df), df['City'].nunique(), f"{df['Year'].min()}-{df['Year'].max()}",
                f"{df['VIIRS'].median():.2f}", f"{compute_correlations(df, 'Park_km2', 'PM25')[0]:.3f}",
                str(PYGAM_AVAILABLE), str(TORCH_AVAILABLE),
                str(len([k for k in gamm_analyzer.models.keys()])),
                str(len(dl_analyzer.models))
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        # GAMM Results
        gamm_summary = []
        for key, value in gamm_results.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if not isinstance(v, (np.ndarray, list)):
                        gamm_summary.append({'Model': key, 'Parameter': k, 'Value': str(v)})
        if gamm_summary:
            pd.DataFrame(gamm_summary).to_excel(writer, sheet_name='GAMM_Results', index=False)

        # Blue-Green Comparison
        if 'blue_green' in dl_results and len(dl_results['blue_green']) > 0:
            dl_results['blue_green'].to_excel(writer, sheet_name='BlueGreen', index=False)

        # Threshold Analysis
        if 'threshold_analysis' in dl_results:
            dl_results['threshold_analysis']['threshold_curve'].to_excel(
                writer, sheet_name='Threshold_Curve', index=False)

        # GAM vs Linear Comparison
        if 'gam_vs_linear' in gamm_results:
            comp = gamm_results['gam_vs_linear']
            comp_df = pd.DataFrame({
                'Model': ['Linear', 'GAM'],
                'R2': [comp['linear']['r2'], comp['gam']['r2']],
                'AIC': [comp['linear']['aic'], comp['gam']['aic']],
            })
            comp_df.to_excel(writer, sheet_name='GAM_vs_Linear', index=False)

        # Model Comparison Summary
        model_comparison = {
            'Method': ['Linear', 'Polynomial', 'GAM', 'GAMM (Mixed)', 'Neural Network', 'LSTM'],
            'Type': ['Parametric', 'Parametric', 'Semi-parametric', 'Semi-parametric', 'Non-parametric',
                     'Non-parametric'],
            'Interpretability': ['High', 'Medium', 'High', 'High', 'Low', 'Low'],
            'Nonlinearity': ['No', 'Limited', 'Yes (smooth)', 'Yes (smooth)', 'Yes (flexible)', 'Yes (temporal)'],
            'Uncertainty_Quantification': ['Yes', 'Yes', 'Yes (built-in)', 'Yes (built-in)', 'Requires Bootstrap',
                                           'Requires Bootstrap'],
            'Random_Effects': ['No', 'No', 'No', 'Yes', 'No', 'No'],
        }
        pd.DataFrame(model_comparison).to_excel(writer, sheet_name='Method_Comparison', index=False)

    print_status(f"Excel exported: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    print("\n" + "█" * 70)
    print("█" + " " * 5 + "GBI Analysis - GAMM + Deep Learning Integrated Version" + " " * 5 + "█")
    print("█" * 70)

    # Initialize analyzers
    gamm_analyzer = GAMMAnalyzer()
    dl_analyzer = DeepLearningAnalyzer()

    print(f"\n  Available Methods:")
    print(f"    GAMM (PyGAM): {PYGAM_AVAILABLE}")
    print(f"    GAMM (Statsmodels): {STATSMODELS_AVAILABLE}")
    print(f"    Deep Learning (PyTorch): {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"    PyTorch Device: {dl_analyzer.device}")

    # ==================== Data Loading ====================
    df = None
    data_source = "UNKNOWN"

    print_header("Checking Data Sources")
    paths_status = {}
    for name, path in PATHS.items():
        exists = path.exists()
        paths_status[name] = exists
        print_status(f"{name}: {path}", exists)

    # Try loading real data
    essential_paths = ['viirs', 'co2', 'pm25']
    if all(paths_status.get(p, False) for p in essential_paths):
        print("\n  🔄 Attempting to load REAL data...")
        try:
            df = load_all_real_data()
            key_cols = ['VIIRS', 'CO2_100Mt', 'PM25', 'Park_km2']
            valid_counts = {col: df[col].notna().sum() for col in key_cols if col in df.columns}
            if all(v > 10 for v in valid_counts.values()):
                data_source = "★ REAL DATA ★"
                print(f"\n  ✓✓✓ Successfully loaded REAL data! ✓✓✓")
            else:
                print(f"\n  ⚠️ Real data incomplete, falling back to simulated...")
                df = None
        except Exception as e:
            print_status(f"Real data loading failed: {e}", False)
            df = None

    # Try loading from CSV
    if df is None and MERGED_DATA_PATH.exists():
        print(f"\n  🔄 Trying merged CSV: {MERGED_DATA_PATH}")
        try:
            df = pd.read_csv(MERGED_DATA_PATH)
            if 'City_EN' not in df.columns and 'City' in df.columns:
                df['City_EN'] = df['City']
            data_source = "MERGED CSV"
            print_status("Merged CSV loaded", True)
        except Exception as e:
            print_status(f"CSV failed: {e}", False)
            df = None

    # Use simulated data
    if df is None:
        df = generate_simulated_data()
        data_source = "⚠️ SIMULATED DATA ⚠️"

    print(f"\n  {'=' * 50}")
    print(f"  DATA SOURCE: {data_source}")
    print(f"  {'=' * 50}")

    # Preprocess
    df, viirs_median = preprocess_data(df)

    # ==================== Run GAMM Analysis ====================
    gamm_results = run_gamm_analysis(df, gamm_analyzer)

    # ==================== Run Deep Learning Analysis ====================
    dl_results = run_deep_learning_analysis(df, dl_analyzer)

    # ==================== Generate Figures ====================
    print_header("Generating Figures")

    try:
        create_fig1_gamm_analysis(df, gamm_results, dl_results, viirs_median,
                                  gamm_analyzer, dl_analyzer,
                                  OUTPUT_DIR / "Fig1_GAMM_Analysis.png")
    except Exception as e:
        print_status(f"Fig1 failed: {e}", False)
        import traceback
        traceback.print_exc()

    try:
        create_fig2_blue_green_analysis(df, dl_results, gamm_analyzer,
                                        OUTPUT_DIR / "Fig2_BlueGreen_Comparison.png")
    except Exception as e:
        print_status(f"Fig2 failed: {e}", False)

    try:
        create_fig3_threshold_analysis(df, gamm_results, dl_results,
                                       gamm_analyzer, dl_analyzer,
                                       OUTPUT_DIR / "Fig3_Threshold_Analysis.png")
    except Exception as e:
        print_status(f"Fig3 failed: {e}", False)

    try:
        create_fig4_city_analysis(df, viirs_median, gamm_analyzer, dl_analyzer,
                                  OUTPUT_DIR / "Fig4_City_Analysis.png")
    except Exception as e:
        print_status(f"Fig4 failed: {e}", False)

    try:
        create_fig5_method_comparison(df, gamm_results, dl_results,
                                      gamm_analyzer, dl_analyzer,
                                      OUTPUT_DIR / "Fig5_Method_Comparison.png")
    except Exception as e:
        print_status(f"Fig5 failed: {e}", False)

    # ==================== Export Results ====================
    try:
        export_results_to_excel(df, gamm_results, dl_results,
                                gamm_analyzer, dl_analyzer,
                                OUTPUT_DIR / "GBI_GAMM_DL_Results.xlsx")
    except Exception as e:
        print_status(f"Excel export failed: {e}", False)

    # Save processed data
    df.to_csv(OUTPUT_DIR / 'processed_data.csv', index=False, encoding='utf-8-sig')
    print_status(f"Processed data saved: {OUTPUT_DIR / 'processed_data.csv'}")

    # ==================== Summary ====================
    print("\n" + "█" * 70)
    print("█" + " " * 28 + "COMPLETE!" + " " * 28 + "█")
    print("█" * 70)

    print(f"\n  Output Directory: {OUTPUT_DIR}")
    print(f"\n  DATA SOURCE: {data_source}")

    print(f"\n  【KEY FINDINGS - GAMM】")
    if 'gam_park_pm25' in gamm_results:
        print(f"    GAM Park→PM2.5 R²: {gamm_results['gam_park_pm25']['pseudo_r2']:.4f}")
    if 'gam_vs_linear' in gamm_results:
        print(
            f"    Nonlinearity Significant: {gamm_results['gam_vs_linear']['improvement']['nonlinearity_significant']}")
    if 'gam_threshold' in gamm_results:
        print(f"    GAM Detected Threshold: {gamm_results['gam_threshold']['threshold']:.1f} km²")
    if 'gamm_pm25' in gamm_results:
        print(f"    GAMM (Mixed) AIC: {gamm_results['gamm_pm25']['aic']:.1f}")

    print(f"\n  【KEY FINDINGS - Deep Learning】")
    r_pm, _, sig = compute_correlations(df, 'Park_km2', 'PM25')
    print(f"    Park → PM2.5 Correlation: r = {r_pm:.3f}{sig}")
    if 'dl_threshold_pm25' in dl_results:
        print(f"    DL Detected Threshold: {dl_results['dl_threshold_pm25']['threshold']:.1f} km²")

    print(f"\n  【MODELS TRAINED】")
    print(f"    GAMM Models: {len(gamm_analyzer.models)}")
    for name in gamm_analyzer.models.keys():
        print(f"      - {name}")
    print(f"    DL Models: {len(dl_analyzer.models)}")
    for name in dl_analyzer.models.keys():
        print(f"      - {name}")

    return df, gamm_results, dl_results, gamm_analyzer, dl_analyzer


if __name__ == "__main__":
    df, gamm_results, dl_results, gamm_analyzer, dl_analyzer = main()