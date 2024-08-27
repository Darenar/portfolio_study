from typing import List

import numpy as np
import datetime as dt
import pandas as pd
import datetime as dt
from scipy.stats import norm


from dataclasses import dataclass


@dataclass
class PortfolioStateStats:
    date: dt.date
    portfolio_value: float
    weights: pd.Series
    shares: pd.Series
    values: pd.Series
    prices: pd.Series
    returns: pd.Series
    portfolio_return: float = 0
    transaction_cost: float = 0
    delta_weights: float = 0  
    sharpe: float = 0
    prev_ret: float = 0
    prev_volatility: float = 0
    portfolio_risk: float = 0
    estimated_mean_returns: np.ndarray = None
    estimated_cov_matrix: np.ndarray = None

    @classmethod
    def initialize(cls, portfolio_value: float, date: dt.datetime, asset_names: List[str]) -> 'PortfolioStateStats':
        empty_series = pd.Series(np.array([0] * len(asset_names)), index=asset_names)
        return cls(
            date=date,
            portfolio_value=portfolio_value,
            weights=empty_series,
            shares=empty_series,
            prices=empty_series,
            returns=empty_series,
            values=empty_series
        )


def calculate_monthly_price(daily_price_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the dataframe with daily prices (dates - indexes, assets - columns) - calculate monthly price.
    """
    # Make sure the index is in the datetime format
    daily_price_df.index = pd.to_datetime(daily_price_df.index)
    monthly_price_df = daily_price_df.groupby(
        pd.Grouper(level=daily_price_df.index.name, freq='m', label='left')).first()
    
    monthly_price_df.index = monthly_price_df.index + dt.timedelta(days=1)
    return monthly_price_df


def calculate_portfolio_state_stats(date: dt.datetime, state_weights: np.ndarray, 
                                    state_prices: np.ndarray, state_returns: pd.Series, 
                                    previous_state: PortfolioStateStats, transaction_cost: float = 0, all_returns= None, 
                                    estimated_mean_returns=None, estimated_cov_matrix = None) -> PortfolioStateStats:
    
    curr_state_value = state_weights * previous_state.portfolio_value
    volume_buy  = np.maximum(curr_state_value - previous_state.values, 0)
    volume_sell = np.maximum(previous_state.values - curr_state_value, 0)
    curr_state_transaction_cost = (volume_buy + volume_sell).sum() * transaction_cost
    curr_portfolio_return = (state_weights*state_returns).sum()
    prev_volatility = calculate_portolio_risk(weights=state_weights, cov_matrix=state_prices.pct_change().cov())
    prev_ret = calculate_portolio_returns(weights=state_weights, mean_returns=state_prices.pct_change().mean())
    sharpe = prev_ret / prev_volatility

    port_risk = all_returns.dot(state_weights).std()
    return PortfolioStateStats(
        date=date,
        weights=state_weights,
        shares=previous_state.portfolio_value * state_weights / previous_state.prices,
        prices=state_prices,
        returns=state_returns,
        portfolio_return=(state_weights*state_returns).sum(),
        values=curr_state_value,
        sharpe=sharpe,
        prev_ret=prev_ret,
        prev_volatility=prev_volatility,
        transaction_cost=curr_state_transaction_cost,
        delta_weights=np.sum(np.abs(state_weights - previous_state.weights)),
        portfolio_value=(previous_state.portfolio_value - curr_state_transaction_cost) * (1 + curr_portfolio_return),
        portfolio_risk=port_risk,
        estimated_mean_returns=estimated_mean_returns,
        estimated_cov_matrix = estimated_cov_matrix
    )


def get_time_factor(freq: str)->float:
    if freq=='daily':
        return 252
    elif freq=='monthly':
        return 12
    else:
        raise NotImplementedError(f"Unknown parameters freq = {freq} has been provided to the object.")


def calculate_portolio_returns(mean_returns: pd.DataFrame, factor: int = 1, weights: pd.DataFrame=None) -> float:
    if weights is None:
        return ((1 + mean_returns) ** factor) - 1.
    return np.sum(mean_returns * factor * weights)


def calculate_portolio_risk(cov_matrix: pd.DataFrame, factor: int = 1, weights: pd.DataFrame=None) -> float:
    if weights is None:
        std_vector = pd.Series(np.sqrt(np.diag(cov_matrix))[0], index=cov_matrix.index)
        return std_vector * np.sqrt(factor)
    tmp= np.dot(weights.T, np.dot(cov_matrix * factor, weights))
    if tmp <= 0:
        tmp = 1e-20  # set std to a tiny number
    return np.sqrt(tmp)


def get_sorted_portolio(factor_series: pd.Series, long_quantile: float, short_quantile: float = None) -> pd.Series:
    """
    Calculate weights given a factor series. 
    Put the top long-quantile assets into a Long position and the bottom short_quantile to the short.
    Parameters
    ----------
    factor_series : pd.Series
        Pandas series with the factor values and asset_name as index
    long_quantile : float
        All assets above this quantile will be put on long position
    short_quantile : float, optional
        All assets below this quantile will be put on short position, by default None

    Returns
    -------
    pd.Series
        Pandas series with weights as values and asset_names as index 
    """
    sort_weights = factor_series.copy()
    sort_weights[:] = 0
    
    # Long
    long_q_val = factor_series.quantile(long_quantile)
    n_long_assets = factor_series[factor_series>=long_q_val].shape[0]
    sort_weights.loc[factor_series>=long_q_val] = 1 / n_long_assets

    # Short
    if short_quantile is not None:
        short_q_val = factor_series.quantile(short_quantile)
        n_short_assets = factor_series[factor_series<=short_q_val].shape[0]
        sort_weights.loc[factor_series<=short_q_val] = -1 / n_short_assets
    return sort_weights


class CentroidSort:

    @staticmethod
    def get_centroid(n_elements: int) -> List[float]:
        """
        Generate a sort vector from the paper R. Almgren and N. Chriss, "Portfolios from Sorts" 
    
        Parameters
        ----------
        n_elements : int
            Number of elements in the sort

        Returns
        -------
        List[float]
            List with centroid values
        """
        alpha = 0.4424  - 0.1185 * (n_elements ** (-0.21))
        return [norm.ppf(
            (n_elements + 1 - i - alpha) / (n_elements - (2 * alpha) + 1)
        ) for i in range(1, n_elements+1)]
    

    def get_signed_centroid(self, factor_series: pd.Series, simulations: int) -> List[float]:
        factor_series.sort_values(ascending=False, inplace=True)
        pos_assets = factor_series[factor_series>=0].shape[0]
        iter_res_list = list()
        for _ in range(simulations):
            sim_iter_res = sorted(np.random.normal(size=factor_series.shape[0]), reverse=True)
            sim_iter_res[:pos_assets] = np.abs(sim_iter_res[:pos_assets])
            sim_iter_res[pos_assets:] = -np.abs(sim_iter_res[pos_assets:])
            sim_iter_res = sorted(sim_iter_res, reverse=True)
            iter_res_list.append(sim_iter_res)
        return pd.Series(np.mean(iter_res_list, axis=0), index=factor_series.index)
    
    def get_ranking(self, factor_series: pd.Series, signed: bool = False, simulations: int = 1000, 
                    group_series: pd.Series = None) -> pd.Series:  
        centroid_weights = factor_series.copy()
        centroid_weights = centroid_weights.sort_values(ascending=False)
        if signed:
            centroid_weights = self.get_signed_centroid(factor_series, simulations=simulations)
        else:
            centroid_weights[:] = self.scale_range(self.get_centroid(centroid_weights.shape[0]))
            # centroid_weights[:] = self.get_centroid(centroid_weights.shape[0])
        centroid_weights = centroid_weights.loc[factor_series.index]
        # Centroid values come in sorted descending order - that's why sort original one ourselves
        return centroid_weights
    
    @staticmethod
    def scale_range(centroid: List[float]) -> List[float]:
        min_centroid_val = min(centroid)
        max_centroid_val = max(centroid)
        return [
            # ((c - min_centroid_val) * 0.1) / (max_centroid_val - min_centroid_val) - 0.05
            ((c - min_centroid_val) * 0.03) / (max_centroid_val - min_centroid_val) - 0.015
            for c in centroid
        ]
