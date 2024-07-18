from typing import List

import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import stats
from scipy.stats import norm

from .base import LogMixin
from .portfolio_optimizer import run_optimization, get_efficient_portfolio
from .utils import calculate_portfolio_state_stats, PortfolioStateStats, calculate_monthly_price



class PortfolioStudy(LogMixin):
    def __init__(self, estimation_period: int, prediction_period: int = 1, cash_start: int = 100_000, 
                 objective: str = 'min_variance', optimization_cost: float = 0.001,
                 transaction_cost: float = 0.001):
        self.estimation_period = estimation_period
        self.prediction_period = prediction_period
        self.cash_start = cash_start
        self.objective = objective
        self.optimization_cost = optimization_cost
        self.transaction_cost = transaction_cost

    def run_study(self, daily_price_df: pd.DataFrame, target: float = None):
        if daily_price_df.isna().sum().sum():
            raise NotImplementedError(f"""Input Daily data has empty values""")
        # Get monthly prices
        monthly_price_df = calculate_monthly_price(daily_price_df)
        # Get Monthly returns
        monthly_returns_df = monthly_price_df.pct_change().dropna(axis=0)
        monthly_price_df = monthly_price_df.iloc[1:]
        n_months, n_assets = monthly_returns_df.shape
        n_study_runs = (n_months-self.estimation_period) / self.prediction_period
        if n_study_runs <= 1:
            raise ValueError(f"The number of study runs is insufficient. Try setting estimation_period/prediction_period to lower values.")
        self.logger.info(f"""Study will be run based on {n_study_runs} runs and {n_assets} assets.""")

        # Initialize study runs and initial state
        prev_portfolio_state = PortfolioStateStats.initialize(
            portfolio_value=self.cash_start, 
            date=monthly_returns_df.index[self.estimation_period],
            asset_names=monthly_returns_df.columns
        )
        study_runs = [prev_portfolio_state]
        start_index = 0
        TO_DEL_LIST = list()
        TO_DEL_TWO_LIST = list()
        for i in tqdm(range(1, int(n_study_runs) + 1)):
            start_prediction_index = start_index + self.estimation_period
            iter_estimation_df = monthly_returns_df.iloc[start_index:start_prediction_index, :]
            iter_prediction_df = monthly_returns_df.iloc[start_prediction_index:start_prediction_index + self.prediction_period, :]
            iter_start_price_df = monthly_price_df.iloc[start_index:start_prediction_index, :]
            TO_DEL_LIST.append(iter_estimation_df.copy())
            TO_DEL_TWO_LIST.append((iter_estimation_df.copy(), target, prev_portfolio_state.weights, self.optimization_cost))
            # Given estimation returns - optimize and get optimal weights
            curr_weight = get_efficient_portfolio(
                mean_returns=iter_estimation_df.mean(), 
                cov_matrix=iter_estimation_df.cov(), 
                target = target, 
                prev_weights=prev_portfolio_state.weights, 
                cost=self.optimization_cost
            )
            curr_portfolio_state = calculate_portfolio_state_stats(
                date=iter_prediction_df.index[0],
                state_weights=curr_weight,
                state_prices=iter_start_price_df,
                state_returns=iter_prediction_df.mean(),
                previous_state=prev_portfolio_state,
                transaction_cost=self.transaction_cost
            )
            study_runs.append(curr_portfolio_state)
            prev_portfolio_state = curr_portfolio_state
            start_index += 1
        return study_runs, TO_DEL_LIST, TO_DEL_TWO_LIST
    
    def evaluate(self, study_runs: List[PortfolioStateStats], freq: str='monthly', risk_free_rate: float=0.2) -> dict:
        performance_stats = dict()
        factor = None
        if freq == 'monthly':
            factor = 12
        elif freq == 'daily':
            factor = 252
        else:
            raise ValueError(f"Unknown parameter freq has been provided: {freq}")
        performance_stats['turnover'] = np.mean([v.delta_weights for v in study_runs]) * factor

        portfolio_values_over_time = pd.Series(
            [v.portfolio_value for v in study_runs], 
            index=[v.date for v in study_runs])
        
        port_returns = portfolio_values_over_time.pct_change().dropna()
        log_port_returns = np.log(portfolio_values_over_time).diff().dropna()

        performance_stats['annualized_skew'] = (factor * log_port_returns).skew()
        performance_stats['annualized_kurtosis'] = stats.kurtosis((factor * log_port_returns).to_list(), fisher=False)
        performance_stats['cumulative_returns'] = ((port_returns+1).prod(skipna=True) - 1) * 100
        
        annual_returns = (port_returns+1).resample('A').prod() - 1.0
        performance_stats['annual_arithmetic_returns'] = annual_returns.mean() * 100
        performance_stats['annual_geometric_returns'] = (
            ((annual_returns + 1).prod()) ** (1.0 / annual_returns.shape[0]) - 1.0) * 100

        # Sharpe
        performance_stats['annualized_mean'] = (((port_returns+1).prod(skipna=True) ** (factor/port_returns.shape[0])) - 1.0) * 100
        performance_stats['annualized_std'] = ((port_returns.std(ddof=0, skipna=True)) * np.sqrt(factor)) * 100
        performance_stats['sharpe_ratio'] = (
            (performance_stats['annualized_mean'] - risk_free_rate) / (performance_stats['annualized_std']))
        performance_stats['monthly_var_quantile'] = port_returns.quantile(0.05)
        return performance_stats
