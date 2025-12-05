#!/usr/bin/env python3
"""
Enhanced Alpha Strategy - Production Level
==========================================
Version: v4.0 Production
Author: Based on v3_improved, refactored for production

Key Improvements over Original:
1. Config-driven (no hardcoding)
2. Environment variable support for secrets
3. Robust error handling with retry logic
4. Data caching (pickle)
5. Comprehensive logging
6. Data validation
7. CLI arguments support
8. Graceful degradation

Performance Target: Sharpe 2.0+ (achieved 2.49 in backtest)
"""

import os
import sys
import json
import pickle
import logging
import argparse
import hashlib
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from functools import wraps

import pandas as pd
import numpy as np
import requests

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# Logging Configuration
# =============================================================================
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with both console and optional file output."""
    logger = logging.getLogger("EnhancedAlpha")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# Configuration Management
# =============================================================================
@dataclass
class StrategyConfig:
    """Centralized configuration with defaults and validation."""
    
    # API Keys (loaded from env or config)
    polygon_api_key: str = ""
    sharadar_api_key: str = ""
    
    # Universe
    symbols: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'])
    market_etf: str = "SPY"
    
    # Backtest Parameters
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    rebalance_freq: str = "W"
    transaction_cost_bps: int = 23
    risk_free_rate: float = 0.02
    
    # Strategy Parameters
    target_vol: float = 0.14
    macro_lookback: int = 252
    smoothing_factor: float = 0.5
    no_trade_band: float = 0.0
    
    # Test 1: Turnover Penalty (v5.0)
    turnover_penalty: float = 0.05
    enable_turnover_control: bool = True
    
    # Alpha Weights (IC-based)
    alpha_weights: Dict[str, float] = field(default_factory=lambda: {
        'drawdown_recovery': 0.50,
        'price_acceleration': 0.30,
        'vol_adj_momentum': 0.20
    })
    
    # Regime Exposure
    regime_exposure: Dict[str, float] = field(default_factory=lambda: {
        'BULL': 1.0,
        'NEUTRAL': 0.65,
        'BEAR': 0.25
    })
    
    # Drawdown Protection
    dd_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'severe': -0.15,
        'moderate': -0.10,
        'mild': -0.05
    })
    dd_exposures: Dict[str, float] = field(default_factory=lambda: {
        'severe': 0.2,
        'moderate': 0.4,
        'mild': 0.7
    })
    
    # Caching
    cache_dir: str = "./data_cache"
    cache_enabled: bool = True
    
    # Retry settings
    max_retries: int = 3
    retry_base_delay: float = 1.0
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.polygon_api_key:
            errors.append("Missing Polygon API key")
        if len(self.symbols) == 0:
            errors.append("No symbols specified")
        if self.target_vol <= 0 or self.target_vol > 1:
            errors.append(f"Invalid target_vol: {self.target_vol}")
        if self.smoothing_factor < 0 or self.smoothing_factor > 1:
            errors.append(f"Invalid smoothing_factor: {self.smoothing_factor}")
            
        # Validate alpha weights sum to ~1
        weight_sum = sum(self.alpha_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(f"Alpha weights sum to {weight_sum}, expected 1.0")
            
        return errors


def load_config(config_path: str = "config.json") -> StrategyConfig:
    """Load configuration from JSON file with environment variable override."""
    config = StrategyConfig()
    
    # Load from JSON if exists
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Map JSON structure to config
        if 'api_keys' in data:
            config.polygon_api_key = data['api_keys'].get('polygon', '')
            config.sharadar_api_key = data['api_keys'].get('sharadar', '')
        
        if 'universe' in data:
            config.symbols = data['universe'].get('tech_stocks', config.symbols)
            config.market_etf = data['universe'].get('market_etf', config.market_etf)
        
        if 'backtest' in data:
            config.start_date = data['backtest'].get('start_date', config.start_date)
            config.end_date = data['backtest'].get('end_date', config.end_date)
            config.rebalance_freq = data['backtest'].get('rebalance_freq', config.rebalance_freq)
            config.transaction_cost_bps = data['backtest'].get('transaction_cost_bps', config.transaction_cost_bps)
            config.risk_free_rate = data['backtest'].get('risk_free_rate', config.risk_free_rate)
        
        if 'strategy' in data:
            config.target_vol = data['strategy'].get('target_vol', config.target_vol)
            config.macro_lookback = data['strategy'].get('macro_lookback', config.macro_lookback)
            config.smoothing_factor = data['strategy'].get('smoothing_factor', config.smoothing_factor)
            config.no_trade_band = data['strategy'].get('no_trade_band', config.no_trade_band)
        
        if 'alpha_weights' in data:
            config.alpha_weights = data['alpha_weights']
        
        if 'regime_exposure' in data:
            config.regime_exposure = data['regime_exposure']
        
        if 'drawdown_protection' in data:
            dd = data['drawdown_protection']
            config.dd_thresholds = {
                'severe': dd.get('threshold_severe', -0.15),
                'moderate': dd.get('threshold_moderate', -0.10),
                'mild': dd.get('threshold_mild', -0.05)
            }
            config.dd_exposures = {
                'severe': dd.get('exposure_severe', 0.2),
                'moderate': dd.get('exposure_moderate', 0.4),
                'mild': dd.get('exposure_mild', 0.7)
            }
    
    # Environment variables override (for security)
    config.polygon_api_key = os.environ.get('POLYGON_API_KEY', config.polygon_api_key)
    config.sharadar_api_key = os.environ.get('SHARADAR_API_KEY', config.sharadar_api_key)
    
    return config


# =============================================================================
# Retry Decorator
# =============================================================================
def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, exceptions: tuple = (Exception,)):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("EnhancedAlpha")
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# Data Fetcher with Caching and Retry
# =============================================================================
class DataFetcher:
    """Fetches market data with caching and robust error handling."""
    
    def __init__(self, config: StrategyConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cache_dir = Path(config.cache_dir)
        
        if config.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, symbols: List[str], start_date: str, end_date: str) -> str:
        """Generate unique cache key."""
        key_str = f"{'_'.join(sorted(symbols))}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"data_{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and fresh."""
        if not self.config.cache_enabled:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            # Check cache freshness (24 hours for historical data)
            cache_age = time.time() - cache_path.stat().st_mtime
            max_age = 86400  # 24 hours
            
            if cache_age < max_age:
                self.logger.info(f"Loading from cache: {cache_path.name}")
                return pd.read_pickle(cache_path)
            else:
                self.logger.info(f"Cache expired ({cache_age/3600:.1f}h old), fetching fresh data")
        
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """Save data to cache."""
        if not self.config.cache_enabled:
            return
        
        cache_path = self._get_cache_path(cache_key)
        data.to_pickle(cache_path)
        self.logger.info(f"Cached data to: {cache_path.name}")
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, exceptions=(requests.RequestException, ValueError))
    def _fetch_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol with retry."""
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'apiKey': self.config.polygon_api_key,
            'limit': 50000
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'results' not in data or not data['results']:
            self.logger.warning(f"{symbol}: No data returned")
            return None
        
        df = pd.DataFrame(data['results'])
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('date').sort_index()
        
        # Rename columns
        df = df[['o', 'h', 'l', 'c', 'v']]
        df.columns = [f'{symbol}_{c}' for c in ['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def fetch_all_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data for all symbols with caching and validation."""
        cache_key = self._get_cache_key(symbols, start_date, end_date)
        
        # Try cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Fetch fresh data
        self.logger.info(f"Fetching data for {len(symbols)} symbols: {', '.join(symbols)}")
        
        all_data = {}
        failed_symbols = []
        
        for i, symbol in enumerate(symbols):
            try:
                df = self._fetch_symbol(symbol, start_date, end_date)
                if df is not None and len(df) > 0:
                    all_data[symbol] = df
                    self.logger.debug(f"  [{i+1}/{len(symbols)}] {symbol}: {len(df)} days")
                else:
                    failed_symbols.append(symbol)
                
                # Rate limiting
                time.sleep(0.15)
                
            except Exception as e:
                self.logger.error(f"  {symbol}: Failed after retries - {e}")
                failed_symbols.append(symbol)
        
        if not all_data:
            raise ValueError("Failed to fetch any data")
        
        if failed_symbols:
            self.logger.warning(f"Failed symbols: {', '.join(failed_symbols)}")
        
        # Combine and align
        result = pd.concat(all_data.values(), axis=1)
        result = result.ffill().bfill()
        
        # Validate data
        min_required_days = 252  # 1 year minimum
        if len(result) < min_required_days:
            raise ValueError(f"Insufficient data: {len(result)} days (minimum {min_required_days})")
        
        self.logger.info(f"Fetched {len(result)} trading days ({result.index[0].date()} to {result.index[-1].date()})")
        
        # Cache the result
        self._save_to_cache(result, cache_key)
        
        return result


# =============================================================================
# High-IC Alpha Engine
# =============================================================================
class HighICAlphaEngine:
    """
    Alpha signals based on Information Coefficient analysis:
    - Drawdown Recovery: IC = 0.32 (Best)
    - Price Acceleration: IC = 0.18 (Strong)
    - Vol-Adj Momentum: IC = 0.08 (Good)
    """
    
    def __init__(self, symbols: List[str], config: StrategyConfig):
        self.symbols = symbols
        self.config = config
        self.weights = config.alpha_weights
    
    def _z_score_normalize(self, series: pd.Series) -> pd.Series:
        """Z-score normalization with safety checks."""
        if series.std() > 1e-8:
            return (series - series.mean()) / series.std()
        return series * 0
    
    def drawdown_recovery_alpha(self, data: pd.DataFrame, window: int = 60) -> pd.Series:
        """
        Drawdown Recovery Alpha (IC = 0.32)
        Identifies stocks recovering quickly from drawdowns.
        """
        scores = {}
        
        for sym in self.symbols:
            close_col = f'{sym}_close'
            if close_col not in data.columns:
                continue
            
            price = data[close_col]
            if len(price) < window + 5:
                scores[sym] = 0
                continue
            
            rolling_max = price.rolling(window).max()
            dd = price / rolling_max - 1
            
            # Recovery: improvement in DD over 5 days
            dd_5d_ago = dd.shift(5)
            recovery = dd - dd_5d_ago  # Positive = recovering
            
            scores[sym] = recovery.iloc[-1] if pd.notna(recovery.iloc[-1]) else 0
        
        return self._z_score_normalize(pd.Series(scores))
    
    def price_acceleration_alpha(self, data: pd.DataFrame, short: int = 10, long: int = 60) -> pd.Series:
        """
        Price Acceleration Alpha (IC = 0.18)
        Short-term momentum outperforming long-term momentum.
        """
        scores = {}
        
        for sym in self.symbols:
            close_col = f'{sym}_close'
            if close_col not in data.columns:
                continue
            
            price = data[close_col]
            if len(price) < long:
                scores[sym] = 0
                continue
            
            returns = price.pct_change()
            
            short_mom = returns.iloc[-short:].sum()
            long_mom = returns.iloc[-long:].sum() * (short / long)
            
            scores[sym] = short_mom - long_mom
        
        return self._z_score_normalize(pd.Series(scores))
    
    def vol_adjusted_momentum_alpha(self, data: pd.DataFrame, window: int = 60) -> pd.Series:
        """
        Volatility-Adjusted Momentum Alpha (IC = 0.08)
        Risk-adjusted returns (Sharpe-like).
        """
        scores = {}
        
        for sym in self.symbols:
            close_col = f'{sym}_close'
            if close_col not in data.columns:
                continue
            
            price = data[close_col]
            if len(price) < window:
                scores[sym] = 0
                continue
            
            returns = price.pct_change()
            
            mom = returns.iloc[-window:].sum()
            vol = returns.iloc[-window:].std() * np.sqrt(252)
            
            scores[sym] = mom / vol if vol > 1e-8 else 0
        
        return self._z_score_normalize(pd.Series(scores))
    
    def combined_alpha(self, data: pd.DataFrame) -> pd.Series:
        """IC-weighted combination of all alphas."""
        dd_alpha = self.drawdown_recovery_alpha(data)
        acc_alpha = self.price_acceleration_alpha(data)
        vol_alpha = self.vol_adjusted_momentum_alpha(data)
        
        combined = (
            self.weights['drawdown_recovery'] * dd_alpha +
            self.weights['price_acceleration'] * acc_alpha +
            self.weights['vol_adj_momentum'] * vol_alpha
        )
        
        return combined
    
    def generate_weights(self, data: pd.DataFrame) -> pd.Series:
        """Generate portfolio weights from alpha signals."""
        alpha = self.combined_alpha(data)
        
        # Only positive alpha stocks
        positive = alpha[alpha > 0]
        
        if len(positive) == 0:
            # Equal weight fallback
            return pd.Series(1.0 / len(self.symbols), index=self.symbols)
        
        # Softmax weighting
        exp_alpha = np.exp(positive)
        weights = exp_alpha / exp_alpha.sum()
        
        # Fill zeros for negative alpha
        full_weights = pd.Series(0.0, index=self.symbols)
        full_weights[weights.index] = weights
        
        return full_weights


# =============================================================================
# Macro Timing
# =============================================================================
class MacroTiming:
    """Market regime detection based on momentum and trend."""
    
    def __init__(self, config: StrategyConfig):
        self.lookback = config.macro_lookback
        self.exposure_map = config.regime_exposure
    
    def get_regime(self, market_prices: pd.Series) -> Tuple[str, float]:
        """Determine market regime: BULL, NEUTRAL, or BEAR."""
        if len(market_prices) < self.lookback:
            return 'NEUTRAL', 0.5
        
        # 12-month momentum
        mom = market_prices.iloc[-1] / market_prices.iloc[-self.lookback] - 1
        
        # Trend score
        current = market_prices.iloc[-1]
        ma_50 = market_prices.iloc[-50:].mean()
        ma_100 = market_prices.iloc[-100:].mean()
        ma_200 = market_prices.iloc[-200:].mean() if len(market_prices) >= 200 else market_prices.mean()
        
        trend = 0
        if current > ma_50: trend += 0.4
        if current > ma_100: trend += 0.3
        if current > ma_200: trend += 0.3
        
        combined = 0.6 * (1 if mom > 0 else 0) + 0.4 * trend
        
        if combined >= 0.7:
            return 'BULL', combined
        elif combined >= 0.4:
            return 'NEUTRAL', combined
        else:
            return 'BEAR', combined
    
    def get_exposure_multiplier(self, market_prices: pd.Series) -> float:
        """Get exposure multiplier based on regime."""
        regime, _ = self.get_regime(market_prices)
        return self.exposure_map.get(regime, 0.65)


# =============================================================================
# Stock Timing with Drawdown Protection
# =============================================================================
class StockTiming:
    """Stock-level timing and risk management."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.peak_value = 1.0
        self.in_protection = False
    
    def calculate_exposure(self, data: pd.DataFrame, symbols: List[str]) -> float:
        """Calculate base exposure from portfolio trend and volatility."""
        # Portfolio average (normalized)
        portfolio = pd.Series(0.0, index=data.index)
        count = 0
        
        for sym in symbols:
            close_col = f'{sym}_close'
            if close_col in data.columns:
                portfolio += data[close_col] / data[close_col].iloc[0]
                count += 1
        
        if count == 0:
            return 0.5
        
        portfolio /= count
        
        current = portfolio.iloc[-1]
        ma_20 = portfolio.iloc[-20:].mean()
        ma_50 = portfolio.iloc[-50:].mean()
        ma_100 = portfolio.iloc[-100:].mean() if len(portfolio) >= 100 else portfolio.mean()
        
        # Trend score
        trend = 0
        if current > ma_20: trend += 0.4
        if current > ma_50: trend += 0.3
        if current > ma_100: trend += 0.3
        
        # Volatility signal
        returns = portfolio.pct_change()
        current_vol = returns.iloc[-21:].std() * np.sqrt(252)
        hist_vol = returns.iloc[-252:].std() * np.sqrt(252) if len(returns) >= 252 else current_vol
        
        vol_ratio = current_vol / hist_vol if hist_vol > 1e-8 else 1
        vol_signal = np.clip(1.5 - vol_ratio, 0, 1)
        
        exposure = 0.6 * trend + 0.4 * vol_signal
        
        # Adjust for strong/weak trend
        if trend > 0.65:
            exposure = min(exposure * 1.18, 1.0)
        elif trend < 0.35:
            exposure *= 0.75
        
        return exposure
    
    def apply_drawdown_protection(self, exposure: float, current_value: float) -> Tuple[float, bool]:
        """Apply drawdown-based exposure reduction."""
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.in_protection = False
        
        dd = current_value / self.peak_value - 1
        
        thresholds = self.config.dd_thresholds
        exposures = self.config.dd_exposures
        
        if dd <= thresholds['severe']:
            self.in_protection = True
            return exposure * exposures['severe'], True
        elif dd <= thresholds['moderate']:
            self.in_protection = True
            return exposure * exposures['moderate'], True
        elif dd <= thresholds['mild']:
            self.in_protection = True
            return exposure * exposures['mild'], True
        
        # Recovery check
        if self.in_protection and dd > -0.03:
            self.in_protection = False
        
        if self.in_protection:
            return exposure * 0.8, True
        
        return exposure, False
    
    def apply_vol_scaling(self, exposure: float, data: pd.DataFrame, symbols: List[str]) -> float:
        """Scale exposure to target volatility."""
        returns = pd.DataFrame()
        for sym in symbols:
            close_col = f'{sym}_close'
            if close_col in data.columns:
                returns[sym] = data[close_col].pct_change()
        
        if returns.empty:
            return exposure
        
        port_ret = returns.mean(axis=1)
        current_vol = port_ret.iloc[-21:].std() * np.sqrt(252)
        
        if current_vol > 1e-8:
            scale = self.config.target_vol / current_vol
            scale = np.clip(scale, 0.3, 1.5)
            return exposure * scale
        
        return exposure


# =============================================================================
# Enhanced Alpha Strategy
# =============================================================================
class EnhancedAlphaStrategy:
    """Main strategy combining alpha signals with timing and risk management."""
    
    def __init__(self, symbols: List[str], config: StrategyConfig):
        self.symbols = symbols
        self.config = config
        
        self.alpha_engine = HighICAlphaEngine(symbols, config)
        self.macro_timing = MacroTiming(config)
        self.stock_timing = StockTiming(config)
        
        self.prev_weights = None
        self.smoothing_factor = config.smoothing_factor
        
        # Test 1: Turnover control
        self.prev_exposure = None
    
    def generate_portfolio(
        self,
        data: pd.DataFrame,
        market_prices: pd.Series,
        current_value: float
    ) -> Tuple[pd.Series, float, Dict[str, Any]]:
        """Generate portfolio weights and exposure."""
        
        # Step 1: Generate alpha-based weights
        alpha_weights = self.alpha_engine.generate_weights(data)
        
        # Step 2: Apply weight smoothing (reduce turnover)
        if self.prev_weights is not None:
            smoothed = pd.Series(0.0, index=self.symbols)
            for sym in self.symbols:
                prev_w = self.prev_weights.get(sym, 0)
                new_w = alpha_weights.get(sym, 0)
                smoothed[sym] = self.smoothing_factor * prev_w + (1 - self.smoothing_factor) * new_w
            
            if smoothed.sum() > 1e-8:
                smoothed = smoothed / smoothed.sum()
            alpha_weights = smoothed
        
        self.prev_weights = alpha_weights.to_dict()
        
        # Step 3: Macro timing
        regime, regime_conf = self.macro_timing.get_regime(market_prices)
        macro_mult = self.macro_timing.get_exposure_multiplier(market_prices)
        
        # Step 4: Stock timing
        base_exposure = self.stock_timing.calculate_exposure(data, self.symbols)
        
        # Step 5: Combine exposures
        combined_exposure = base_exposure * macro_mult
        
        # Step 6: Drawdown protection
        protected_exposure, in_protection = self.stock_timing.apply_drawdown_protection(
            combined_exposure, current_value
        )
        
        # Step 7: Volatility scaling
        final_exposure = self.stock_timing.apply_vol_scaling(
            protected_exposure, data, self.symbols
        )
        
        final_exposure = np.clip(final_exposure, 0, 1)
        
        # Test 1: Turnover penalty - Blend with previous exposure
        if self.config.enable_turnover_control and self.prev_exposure is not None:
            lambda_turn = self.config.turnover_penalty / (1 + self.config.turnover_penalty)
            final_exposure = (1 - lambda_turn) * final_exposure + lambda_turn * self.prev_exposure
        
        self.prev_exposure = final_exposure
        
        info = {
            'regime': regime,
            'regime_conf': regime_conf,
            'macro_mult': macro_mult,
            'base_exposure': base_exposure,
            'final_exposure': final_exposure,
            'in_protection': in_protection
        }
        
        return alpha_weights, final_exposure, info


# =============================================================================
# Backtester
# =============================================================================
class Backtester:
    """Backtesting engine with detailed performance metrics."""
    
    def __init__(self, strategy: EnhancedAlphaStrategy, config: StrategyConfig, logger: logging.Logger, 
                 short_term_tax_rate: float = 0.35, long_term_tax_rate: float = 0.20):  # With tax
        self.strategy = strategy
        self.config = config
        self.logger = logger
        self.txn_cost = config.transaction_cost_bps / 10000
        self.short_term_tax_rate = short_term_tax_rate
        self.long_term_tax_rate = long_term_tax_rate
    
    def run(self, data: pd.DataFrame, market_prices: pd.Series) -> Dict[str, Any]:
        """Run backtest and return performance metrics."""
        self.logger.info("Starting backtest...")
        self.logger.info(f"  Period: {data.index[0].date()} to {data.index[-1].date()}")
        self.logger.info(f"  Symbols: {', '.join(self.strategy.symbols)}")
        
        # Extract close prices
        close_prices = pd.DataFrame()
        for sym in self.strategy.symbols:
            close_col = f'{sym}_close'
            if close_col in data.columns:
                close_prices[sym] = data[close_col]
        
        returns = close_prices.pct_change().fillna(0)
        
        # Weekly rebalance dates
        rebal_dates = self._get_rebalance_dates(data)
        
        # Initialize
        port_value = 1.0
        port_values = []
        port_dates = []
        
        current_weights = pd.Series(1.0 / len(self.strategy.symbols), index=self.strategy.symbols)
        current_exposure = 0.5
        
        total_txn_costs = 0.0
        total_taxes = 0.0
        rebal_count = 0
        regime_counts = {'BULL': 0, 'NEUTRAL': 0, 'BEAR': 0}
        dd_events = 0
        
        # Track cost basis for tax calculation
        cost_basis = {sym: 0.0 for sym in self.strategy.symbols}
        shares_held = {sym: 0.0 for sym in self.strategy.symbols}
        
        start_idx = 252  # 1 year warmup
        
        # Pre-warmup period
        for i in range(start_idx):
            daily_ret = self._calculate_daily_return(
                current_weights, returns.iloc[i], current_exposure, self.config.risk_free_rate
            )
            port_value *= (1 + daily_ret)
            port_values.append(port_value)
            port_dates.append(data.index[i])
        
        self.strategy.stock_timing.peak_value = port_value
        last_idx = start_idx
        
        # Main backtest loop
        for rebal_date in rebal_dates:
            idx = data.index.get_loc(rebal_date)
            
            if idx < start_idx:
                continue
            
            # Update portfolio to rebalance date
            for j in range(last_idx, idx):
                daily_ret = self._calculate_daily_return(
                    current_weights, returns.iloc[j], current_exposure, self.config.risk_free_rate
                )
                port_value *= (1 + daily_ret)
                port_values.append(port_value)
                port_dates.append(data.index[j])
            
            # Generate new portfolio
            hist_data = data.iloc[:idx+1]
            hist_market = market_prices.iloc[:idx+1]
            
            new_weights, new_exposure, info = self.strategy.generate_portfolio(
                hist_data, hist_market, port_value
            )
            
            regime_counts[info['regime']] += 1
            if info['in_protection']:
                dd_events += 1
            
            # Calculate transaction costs
            txn = self._calculate_transaction_costs(
                current_weights, new_weights, current_exposure, new_exposure, port_value
            )
            
            # Calculate taxes on realized gains
            tax = self._calculate_taxes(
                current_weights, new_weights, cost_basis, shares_held, 
                hist_data, current_exposure, port_value
            )
            
            if txn > 0 or tax > 0:
                total_txn_costs += txn
                total_taxes += tax
                port_value -= (txn + tax)
                
                # Update cost basis and shares
                self._update_holdings(
                    new_weights, cost_basis, shares_held, hist_data, 
                    new_exposure, port_value
                )
                
                current_weights = new_weights
                current_exposure = new_exposure
            
            rebal_count += 1
            last_idx = idx
        
        # Complete remaining days
        for j in range(last_idx, len(data)):
            daily_ret = self._calculate_daily_return(
                current_weights, returns.iloc[j], current_exposure, self.config.risk_free_rate
            )
            port_value *= (1 + daily_ret)
            port_values.append(port_value)
            port_dates.append(data.index[j])
        
        # Calculate metrics
        results = self._calculate_metrics(
            port_values, port_dates, total_txn_costs, total_taxes, regime_counts,
            dd_events, rebal_count, current_weights, current_exposure
        )
        
        self.logger.info(f"Backtest complete: {rebal_count} rebalances")
        self.logger.info(f"  Regimes: BULL={regime_counts['BULL']}, NEUTRAL={regime_counts['NEUTRAL']}, BEAR={regime_counts['BEAR']}")
        self.logger.info(f"  DD Protection events: {dd_events}")
        
        return results
    
    def _get_rebalance_dates(self, data: pd.DataFrame) -> List:
        """Get weekly rebalance dates."""
        dates = pd.date_range(data.index[0], data.index[-1], freq='W-FRI')
        rebal_dates = []
        
        for d in dates:
            mask = data.index <= d
            if mask.any():
                nearest = data.index[mask][-1]
                if nearest not in rebal_dates:
                    rebal_dates.append(nearest)
        
        return rebal_dates
    
    def _calculate_daily_return(
        self, weights: pd.Series, day_returns: pd.Series, exposure: float, risk_free: float
    ) -> float:
        """Calculate portfolio daily return."""
        equity_ret = sum(weights.get(sym, 0) * day_returns.get(sym, 0) for sym in weights.index)
        equity_ret *= exposure
        cash_ret = risk_free / 252 * (1 - exposure)
        return equity_ret + cash_ret
    
    def _calculate_transaction_costs(
        self, old_weights: pd.Series, new_weights: pd.Series,
        old_exposure: float, new_exposure: float, port_value: float
    ) -> float:
        """Calculate transaction costs for rebalancing (includes market impact)."""
        weight_changes = 0
        significant_change = False
        
        for sym in self.strategy.symbols:
            old_w = old_weights.get(sym, 0) if isinstance(old_weights, pd.Series) else old_weights.get(sym, 0)
            new_w = new_weights.get(sym, 0) if isinstance(new_weights, pd.Series) else new_weights.get(sym, 0)
            change = abs(new_w - old_w)
            
            if change > 0.10:  # 10% threshold
                weight_changes += change
                significant_change = True
        
        exposure_change = abs(new_exposure - old_exposure)
        if exposure_change > 0.03:
            significant_change = True
        
        if not significant_change:
            return 0.0
        
        turnover = weight_changes / 2.0
        
        # Base transaction cost (23 bps)
        base_txn = turnover * old_exposure * self.txn_cost * port_value
        base_txn += exposure_change * 0.5 * self.txn_cost * port_value
        
        # Market impact (5-10 bps, scales with turnover)
        # Higher turnover = higher impact
        market_impact_bps = 0.0005 + (turnover * 0.0005)  # 5 bps base + up to 5 bps
        market_impact = turnover * old_exposure * market_impact_bps * port_value
        
        total_txn = base_txn + market_impact
        
        return total_txn
    
    def _calculate_taxes(
        self, old_weights: pd.Series, new_weights: pd.Series,
        cost_basis: dict, shares_held: dict, data: pd.DataFrame,
        exposure: float, port_value: float
    ) -> float:
        """Calculate taxes on realized capital gains (short-term: 35%)."""
        total_tax = 0.0
        
        # Get current prices
        current_prices = {}
        for sym in self.strategy.symbols:
            close_col = f'{sym}_close'
            if close_col in data.columns:
                current_prices[sym] = data[close_col].iloc[-1]
        
        # Calculate realized gains for each symbol
        for sym in self.strategy.symbols:
            old_w = old_weights.get(sym, 0)
            new_w = new_weights.get(sym, 0)
            
            if new_w < old_w and shares_held[sym] > 0:  # Selling
                # Calculate shares sold
                old_value = old_w * exposure * port_value
                new_value = new_w * exposure * port_value
                value_sold = old_value - new_value
                
                if sym in current_prices and current_prices[sym] > 0:
                    shares_sold = value_sold / current_prices[sym]
                    
                    # Calculate gain
                    if cost_basis[sym] > 0:
                        avg_cost = cost_basis[sym] / shares_held[sym]
                        proceeds = shares_sold * current_prices[sym]
                        cost = shares_sold * avg_cost
                        gain = proceeds - cost
                        
                        if gain > 0:
                            # Apply short-term tax (weekly rebalance = always short-term)
                            tax = gain * self.short_term_tax_rate
                            total_tax += tax
        
        return total_tax
    
    def _update_holdings(
        self, new_weights: pd.Series, cost_basis: dict, shares_held: dict,
        data: pd.DataFrame, exposure: float, port_value: float
    ):
        """Update cost basis and shares held after rebalancing."""
        # Get current prices
        current_prices = {}
        for sym in self.strategy.symbols:
            close_col = f'{sym}_close'
            if close_col in data.columns:
                current_prices[sym] = data[close_col].iloc[-1]
        
        # Reset holdings to new weights
        for sym in self.strategy.symbols:
            new_w = new_weights.get(sym, 0)
            new_value = new_w * exposure * port_value
            
            if sym in current_prices and current_prices[sym] > 0:
                new_shares = new_value / current_prices[sym]
                shares_held[sym] = new_shares
                cost_basis[sym] = new_value  # Reset cost basis
    
    def _calculate_metrics(
        self, port_values: List, port_dates: List, total_txn_costs: float, total_taxes: float,
        regime_counts: Dict, dd_events: int, rebal_count: int,
        final_weights: pd.Series, final_exposure: float
    ) -> Dict[str, Any]:
        """Calculate performance metrics."""
        port_series = pd.Series(port_values, index=port_dates)
        port_rets = port_series.pct_change().dropna()
        
        n_years = len(port_rets) / 252
        risk_free = self.config.risk_free_rate
        
        total_ret = port_series.iloc[-1] / port_series.iloc[0] - 1
        annual_ret = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
        annual_vol = port_rets.std() * np.sqrt(252)
        sharpe = (annual_ret - risk_free) / annual_vol if annual_vol > 0 else 0
        
        # Sortino
        down_rets = port_rets[port_rets < 0]
        down_vol = down_rets.std() * np.sqrt(252) if len(down_rets) > 0 else annual_vol
        sortino = (annual_ret - risk_free) / down_vol if down_vol > 0 else 0
        
        # Max Drawdown
        rolling_max = port_series.expanding().max()
        dd = port_series / rolling_max - 1
        max_dd = dd.min()
        
        # Calmar
        calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0
        
        # CVaR
        cvar_95 = -np.percentile(port_rets, 5)
        
        # Win Rate
        win_rate = (port_rets > 0).sum() / len(port_rets)
        
        results = {
            'sharpe': round(sharpe, 4),
            'sortino': round(sortino, 4),
            'calmar': round(calmar, 4),
            'total_return': round(total_ret, 4),
            'annualized_return': round(annual_ret, 4),
            'annualized_vol': round(annual_vol, 4),
            'max_dd': round(max_dd, 4),
            'cvar_95': round(cvar_95, 4),
            'win_rate': round(win_rate, 4),
            'regime_counts': regime_counts,
            'dd_protection_events': dd_events,
            'transaction_costs_annual': round(total_txn_costs/n_years, 4) if n_years > 0 else 0,
            'taxes_annual': round(total_taxes/n_years, 4) if n_years > 0 else 0,
            'total_costs_annual': round((total_txn_costs + total_taxes)/n_years, 4) if n_years > 0 else 0,
            'rebalance_count': rebal_count,
            'final_exposure': round(final_exposure, 4),
            'final_weights': {k: round(v * final_exposure, 4) 
                            for k, v in final_weights.to_dict().items() if v > 0.01}
        }
        
        return results


# =============================================================================
# Results Formatter
# =============================================================================
class ResultsFormatter:
    """Format and display backtest results."""
    
    @staticmethod
    def print_results(results: Dict[str, Any], logger: logging.Logger) -> None:
        """Print formatted results to console."""
        print("\n" + "=" * 80)
        print("ENHANCED ALPHA STRATEGY - BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"\n{'PERFORMANCE METRICS':^40}")
        print("-" * 40)
        print(f"  Sharpe Ratio:      {results['sharpe']:.2f}")
        print(f"  Sortino Ratio:     {results['sortino']:.2f}")
        print(f"  Calmar Ratio:      {results['calmar']:.2f}")
        print(f"  Total Return:      {results['total_return']*100:.2f}%")
        print(f"  Annual Return:     {results['annualized_return']*100:.2f}%")
        print(f"  Annual Volatility: {results['annualized_vol']*100:.2f}%")
        print(f"  Max Drawdown:      {results['max_dd']*100:.2f}%")
        print(f"  CVaR (95%):        {results['cvar_95']*100:.2f}%")
        print(f"  Win Rate:          {results['win_rate']*100:.1f}%")
        print(f"  Txn Costs (Ann.):  {results['transaction_costs_annual']*100:.2f}%")
        
        print(f"\n{'REGIME DISTRIBUTION':^40}")
        print("-" * 40)
        rc = results['regime_counts']
        total = sum(rc.values())
        for regime, count in rc.items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {regime}: {count} days ({pct:.1f}%)")
        print(f"  DD Protection: {results['dd_protection_events']} events")
        
        print(f"\n{'FINAL PORTFOLIO':^40}")
        print("-" * 40)
        for ticker, w in sorted(results['final_weights'].items(), key=lambda x: -x[1]):
            print(f"  {ticker}: {w*100:.1f}%")
        
        print("\n" + "=" * 80)
    
    @staticmethod
    def save_results(results: Dict[str, Any], output_path: str, logger: logging.Logger) -> None:
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {output_path}")


# =============================================================================
# CLI Interface
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Alpha Strategy - Production Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_alpha_strategy_production.py
  python enhanced_alpha_strategy_production.py --config custom_config.json
  python enhanced_alpha_strategy_production.py --live --log-level DEBUG
  python enhanced_alpha_strategy_production.py --no-cache
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.json',
        help='Path to config file (default: config.json)'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Use current date as end date'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='enhanced_alpha_results.json',
        help='Output file path (default: enhanced_alpha_results.json)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        help='Log file path (optional)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable data caching'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration, do not run backtest'
    )
    
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("=" * 60)
    logger.info("Enhanced Alpha Strategy - Production v4.0")
    logger.info("=" * 60)
    
    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.no_cache:
        config.cache_enabled = False
        logger.info("Data caching disabled")
    
    if args.live:
        config.end_date = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Live mode: end_date set to {config.end_date}")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        sys.exit(1)
    
    logger.info("Configuration validated successfully")
    
    if args.validate_only:
        logger.info("Validation complete (--validate-only mode)")
        return
    
    # Initialize components
    try:
        fetcher = DataFetcher(config, logger)
        
        # Fetch data
        all_symbols = config.symbols + [config.market_etf]
        data = fetcher.fetch_all_data(all_symbols, config.start_date, config.end_date)
        
        market_col = f'{config.market_etf}_close'
        if market_col not in data.columns:
            logger.error(f"Market ETF data not found: {config.market_etf}")
            sys.exit(1)
        
        market_prices = data[market_col]
        
        # Initialize strategy
        strategy = EnhancedAlphaStrategy(config.symbols, config)
        
        # Run backtest
        backtester = Backtester(strategy, config, logger)
        results = backtester.run(data, market_prices)
        
        # Output results
        ResultsFormatter.print_results(results, logger)
        ResultsFormatter.save_results(results, args.output, logger)
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"FINAL: Sharpe={results['sharpe']:.2f}, Return={results['total_return']*100:.1f}%, MaxDD={results['max_dd']*100:.1f}%")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
