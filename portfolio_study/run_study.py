from typing import List

import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import stats
from pypfopt.risk_models import risk_matrix


from .base import LogMixin
from .portfolio_optimizer_package import run_optimization
from .utils import calculate_portfolio_state_stats, PortfolioStateStats, get_sorted_portolio, CentroidSort, get_time_factor



class PortfolioStudy(LogMixin):
    def __init__(self, estimation_period: int, prediction_period: int = 1, cash_start: int = 100_000, 
                 objective: str = 'max_sharpe', optimization_cost: float = 0.001,
                 transaction_cost: float = 0.001, long_quantile: float = 0.75, short_quantile: float = None,
                 max_asset_weight: float = 1., min_asset_weight: float = 0.):
        self.estimation_period = estimation_period
        self.prediction_period = prediction_period
        self.cash_start = cash_start
        self.objective = objective
        self.optimization_cost = optimization_cost
        self.transaction_cost = transaction_cost
        self.long_quantile = long_quantile
        self.short_quantile = short_quantile
        self.max_asset_weight = max_asset_weight
        self.min_asset_weight = min_asset_weight


    def run_study(self, price_df: pd.DataFrame, target: float = None, signal_factor: pd.DataFrame = None, 
                  use_centroid: bool = False, signed: bool = False, max_valid_abs_ret: float = 1):
        if price_df.isna().sum().sum():
            raise NotImplementedError(f"""Input data has empty values""")
        if signal_factor is not None and signal_factor.shape[0] != price_df.shape[0]:
            raise ValueError(f"Provided signal factor is not of the same shape as input prices.")
        # Get returns
        returns_df = price_df.pct_change().dropna(axis=0)
        
        # Remove the first date from the rows
        price_df = price_df.iloc[1:]
        if signal_factor is not None:
            signal_factor = signal_factor.iloc[1:]

        # Make sure that outlier returns are replaced with zeros
        self.logger.info(f"Filtering by {max_valid_abs_ret} MaxReturn")
        returns_df[
            returns_df.abs()>=max_valid_abs_ret
        ] = 0
        
        # Calculate number of runs 
        n_months, n_assets = returns_df.shape
        n_study_runs = (n_months-self.estimation_period) / self.prediction_period
        if n_study_runs <= 1:
            raise ValueError(f"The number of study runs is insufficient. Try setting estimation_period/prediction_period to lower values.")
        self.logger.info(f"""Study will be run based on {n_study_runs} runs and {n_assets} assets.""")

        # Initialize study runs and initial state
        prev_portfolio_state = PortfolioStateStats.initialize(
            portfolio_value=self.cash_start, 
            date=returns_df.index[self.estimation_period],
            asset_names=returns_df.columns
        )
        study_runs = [prev_portfolio_state]
        start_index = 0

        for _ in tqdm(range(1, int(n_study_runs) + 1)):

            start_prediction_index = start_index + self.estimation_period
            iter_estimation_df = returns_df.iloc[start_index:start_prediction_index, :].copy()
            iter_prediction_df = returns_df.iloc[start_prediction_index:start_prediction_index + self.prediction_period, :].copy()
            
            
            iter_start_price_df = price_df.iloc[start_index:start_prediction_index, :]
            estimated_cov_matrix = risk_matrix(iter_estimation_df, method="ledoit_wolf", returns_data=True)
            
            ## TODO delete in the final version
            # estimated_cov_matrix = pd.DataFrame(np.eye(iter_estimation_df.shape[1]), columns=iter_estimation_df.columns, index=iter_estimation_df.columns)
            # mean_returns = (iter_estimation_df.mean() + 1)**21 - 1
            
            mean_returns = iter_estimation_df.mean()

            if self.objective == 'equal':
                curr_weight = pd.Series([1./n_assets] * n_assets, index=iter_estimation_df.columns)
            
            elif self.objective == 'sort':
                if signal_factor is not None:
                    # If signal is provided - sort by it
                    mean_returns = signal_factor.iloc[start_prediction_index-1].copy()
                curr_weight = get_sorted_portolio(
                    mean_returns,
                    long_quantile=self.long_quantile,
                    short_quantile=self.short_quantile,
                )
            else:
                # If here - it is optimization scenario
                if signal_factor is not None:
                    mean_returns = signal_factor.iloc[start_prediction_index-1].copy()
                if use_centroid:
                    # Replace original mean returns with a centroid
                    mean_returns = CentroidSort().get_ranking(mean_returns, signed=signed)
                
                curr_weight = run_optimization(
                    mean_returns=mean_returns,
                    cov_matrix=estimated_cov_matrix,
                    objective_type=self.objective, 
                    target=target,
                    prev_weights=prev_portfolio_state.weights, 
                    cost=self.optimization_cost, 
                    min_asset_weight=self.min_asset_weight, 
                    max_asset_weight=self.max_asset_weight) 
            
            # With weights - calculate stats of that iteration    
            curr_portfolio_state = calculate_portfolio_state_stats(
                date=iter_prediction_df.index[0],
                state_weights=curr_weight,
                state_prices=iter_start_price_df,
                state_returns=(iter_prediction_df+1).product()-1,
                all_returns = iter_prediction_df,
                previous_state=prev_portfolio_state,
                transaction_cost=self.transaction_cost,
                estimated_mean_returns = mean_returns,
                estimated_cov_matrix = estimated_cov_matrix
            )
            study_runs.append(curr_portfolio_state)
            # Replace prev_portfolio (which is used for calcualating turnover, for example)
            prev_portfolio_state = curr_portfolio_state
            start_index += self.prediction_period
        return study_runs
    
    def evaluate(self, study_runs: List[PortfolioStateStats], freq: str='monthly', risk_free_rate: float=0.0) -> dict:
        performance_stats = dict()
        factor = get_time_factor(freq=freq)
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
