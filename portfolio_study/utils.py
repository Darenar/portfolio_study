from typing import List

import numpy as np
import datetime as dt
import pandas as pd
import datetime as dt

from dataclasses import dataclass


@dataclass
class PortfolioStateStats:
    date: dt.date
    portfolio_value: float
    weights: pd.Series
    shares: pd.Series
    values: pd.Series
    prices: pd.Series
    portfolio_return: float = 0
    transaction_cost: float = 0
    delta_weights: float = 0  

    @classmethod
    def initialize(cls, portfolio_value: float, date: dt.datetime, asset_names: List[str]) -> 'PortfolioStateStats':
        empty_series = pd.Series(np.array([0] * len(asset_names)), index=asset_names)
        return cls(
            date=date,
            portfolio_value=portfolio_value,
            weights=empty_series,
            shares=empty_series,
            prices=empty_series,
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
                                    previous_state: PortfolioStateStats, transaction_cost: float = 0) -> PortfolioStateStats:
    
    curr_state_value = state_weights * previous_state.portfolio_value
    volume_buy  = np.maximum(curr_state_value - previous_state.values, 0)
    volume_sell = np.maximum(previous_state.values - curr_state_value, 0)
    curr_state_transaction_cost = (volume_buy + volume_sell).sum() * transaction_cost
    curr_portfolio_return = (state_weights*state_returns).sum()

    return PortfolioStateStats(
        date=date,
        weights=state_weights,
        shares=previous_state.portfolio_value * state_weights / previous_state.prices,
        prices=state_prices,
        portfolio_return=(state_weights*state_returns).sum(),
        values=curr_state_value,
        transaction_cost=curr_state_transaction_cost,
        delta_weights=np.sum(np.abs(state_weights - previous_state.weights)),
        portfolio_value=(previous_state.portfolio_value - curr_state_transaction_cost) * (1 + curr_portfolio_return)
    )
