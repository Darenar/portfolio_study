import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .utils import get_time_factor, calculate_portolio_risk, calculate_portolio_returns


def get_efficient_portfolio(mean_returns: pd.DataFrame, cov_matrix: pd.DataFrame, target: float = None,
                            freq: str = 'monthly', min_asset_weight: float = 0., max_asset_weight: float = 1., 
                            prev_weights: pd.DataFrame = None, 
                            cost: float = None, eps: float = 1e-4) -> pd.Series:
    # Define main params for all portfolios
    port_params = dict(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        freq=freq,
        min_asset_weight=min_asset_weight,
        max_asset_weight=max_asset_weight,
        # prev_weights=prev_weights,
        # cost=cost,
        eps=eps
    )
    factor = get_time_factor(freq)
    #### MinVariance portfolio
    port_weights = run_optimization(objective_type='min_risk', target=None, **port_params)
    port_risk = calculate_portolio_risk(weights=port_weights, cov_matrix=cov_matrix, factor=factor)
    if target <= port_risk:
        print('Target risk is lower than the one from MinVariance portfolio. Use weights from the latter.')
        print(port_risk)
        return port_weights
    
    #### MaxReturn portfolio
    port_weights = run_optimization(objective_type='max_return', target=None, **port_params)
    port_risk = calculate_portolio_risk(weights=port_weights, cov_matrix=cov_matrix, factor=factor)
    # Check what the risk of each asset in the portfolio 
    asset_risks = calculate_portolio_risk(cov_matrix=cov_matrix, factor=factor)
    asset_returns = calculate_portolio_returns(mean_returns=mean_returns, factor=factor)
    max_return_asset_name = asset_returns.idxmax()
    if port_risk < asset_risks[max_return_asset_name]:
        port_risk = asset_risks[max_return_asset_name]
        # Set weight = 1 only to the max return asset
        port_weights[:] = 0
        port_weights[max_return_asset_name] = 1
    if port_risk <= target:
        print('Target risk is greater than the one from MaxReturn portfolio. Use weights from the latter.')
        print(port_risk, target)
        return port_weights
    
    #### MeanVariance with a target
    port_weights = run_optimization(objective_type='max_return', target=target, cost=cost, prev_weights=prev_weights, **port_params)
    return port_weights


def run_optimization(mean_returns: pd.DataFrame, cov_matrix: pd.DataFrame, target: float = None,
                     freq: str = 'monthly', min_asset_weight: float = 0., max_asset_weight: float = 1., 
                     objective_type: str = 'max_return', prev_weights: pd.DataFrame = None, 
                     cost: float = None, eps: float = 1e-4) -> pd.Series:
    
    # Get constraints on the weight magnitudes
    weight_bounds = tuple((min_asset_weight, max_asset_weight) for _ in range(mean_returns.shape[0]))
    # Add sum up constraint of weights
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    factor = get_time_factor(freq)
    
    
    if objective_type == 'max_return':
        logging.getLogger(__name__).info(f"Optimizing by maximum return.")
        objective = lambda w: -calculate_portolio_returns(weights=w, mean_returns=mean_returns, factor=factor)
        if target:
            logging.getLogger(__name__).info(f"Target risk {target} has been provided. Optimizing with the constraint.")
            target_constraint_func = lambda w: calculate_portolio_risk(weights=w, cov_matrix=cov_matrix, factor=factor) - target
            constraints.append({'type': 'eq', 'fun': target_constraint_func})
    elif objective_type == 'min_risk':
        logging.getLogger(__name__).info(f"Optimizing by minimum risk.")
        objective = lambda w: calculate_portolio_risk(weights=w, cov_matrix=cov_matrix, factor=factor)
        if target:
            logging.getLogger(__name__).info(f"Target return {target} has been provided. Optimizing with the constraint.")
            target_constraint_func = lambda w: calculate_portolio_returns(weights=w, mean_returns=mean_returns, factor=factor) - target
            constraints.append({'type': 'eq', 'fun': target_constraint_func})
    elif objective_type == 'max_sharpe':
        logging.getLogger(__name__).info(f"Optimizing by maximizing sharpe ratio.")
        objective = lambda w: (
            -(calculate_portolio_returns(weights=w, mean_returns=mean_returns, factor=factor)) /
            calculate_portolio_risk(weights=w, cov_matrix=cov_matrix, factor=factor)
        )
    else:
        raise NotImplementedError(f"Got unexpected objective type {objective_type}. Should be either min_risk or max_return.")

    # Add transaction costs to the optimization problem if applicable
    if prev_weights is not None and sum(prev_weights) and cost is not None:
        main_objective = lambda w: objective(w) + np.abs(w - prev_weights).sum() * cost / 10000.
    else:
        main_objective = objective

    init_weights = np.array(mean_returns.shape[0] * [1./mean_returns.shape[0]])
    try:
        opt = minimize(main_objective, x0=init_weights, bounds=weight_bounds, constraints=constraints, method="SLSQP")
    except:
        # if SLSQP fails then switch to trust-constr
        opt = minimize(main_objective, x0=init_weights, bounds=weight_bounds, constraints=constraints, method="trust-constr")
    # Turn too small values to zeros and make sure the weights still sum up to 1
    opt = [w if np.abs(w)>=eps else 0 for w in opt['x']]

    return pd.Series(data=np.array([w / np.sum(opt) for w in opt]), index=mean_returns.index)
