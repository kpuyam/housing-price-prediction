"""Processors for the model training step of the worklow."""
import logging
import logging.config
import os.path as op

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    load_dataset,
    register_processor,
    save_pipeline
)
import ta_lib.eda.api as eda
import ta_lib.regression.api as SKLStatsmodelOLS
import ta_lib.reports.api as reports

logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """
    Train a regression model.

    Parameters:
    - context: The context object containing information about the execution environment.
    - params: Additional parameters or configuration settings for scoring the model.

    Returns:
    - None
    
    artifacts_folder: Path to folder to store artifacts
    """
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load training datasets
    train_X = load_dataset(context, "train/housing/features")
    train_y = load_dataset(context, "train/housing/target")
    
    out_plot = eda.get_density_plots(pd.DataFrame(train_X))
    print(out_plot)
    reports.create_report({'univariate': out_plot}, name='production/reports/feature_analysis_univariate')
    reports.feature_analysis(train_X, 'production/reports/feature_analysis_report.html')

    # create reports as needed
    cols = train_X.columns.to_list()
    all_plots = {}
    for ii, col1 in enumerate(cols):
        for jj in range(ii + 1, len(cols)):
            col2 = cols[jj]
            out = eda.get_bivariate_plots(train_X, x_cols=[col1], y_cols=[col2])
            all_plots.update({f'{col2} vs {col1}': out})

    reports.create_report(
        all_plots,
        name='production/reports/feature_analysis_bivariate'
    )
    reports.feature_interactions(
        train_X,
        'production/reports/feature_interaction_report.html'
    )
    reports.data_exploration(
        train_X,
        train_y,
        'production/reports/data_exploration_report.html',
        y_continuous=True
    )
    

    # create training pipeline
    lin_reg_ppln = Pipeline([
        ('linreg_estimator', LinearRegression())
    ])
    lin_reg_ppln.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(lin_reg_ppln, op.abspath(op.join(artifacts_folder, "lin_reg_pipeline.joblib")))
    print("LR done")
    
    dtree_reg_ppln = Pipeline([
        ('dtreereg_estimator', DecisionTreeRegressor())
    ])
    dtree_reg_ppln.fit(train_X, train_y.values.ravel())
    
    save_pipeline(dtree_reg_ppln, op.abspath(op.join(artifacts_folder, "dtree_reg_pipeline.joblib")))
    print("DT done")
    
    forest_reg = RandomForestRegressor(random_state=42)
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    rand_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rand_search.fit(train_X, train_y.values.ravel())
    print("Best params from Randomized Search CV", rand_search.best_params_)
    cvres = rand_search.cv_results_
    for mean_score, params_cv in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params_cv)

    feature_importances = rand_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, train_X.columns), reverse=True)
    final_model_rand = rand_search.best_estimator_
    print("Best estimator for Randomized Search CV: ", final_model_rand)
    save_pipeline(final_model_rand, op.abspath(op.join(artifacts_folder, "rand_search.joblib")))
    print("Rand_search")
    print(params)
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )

    grid_search.fit(train_X, train_y.values.ravel())
    print("Best params for Grid Search CV: ", grid_search.best_params_)
    cvres = grid_search.cv_results_
    for mean_score, params_cv in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params_cv)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, train_X.columns), reverse=True)
    final_model_grid = grid_search.best_estimator_
    print("Best estimator for Grid Search CV: ", final_model_grid)
    save_pipeline(final_model_grid, op.abspath(op.join(artifacts_folder, "grid_search.joblib")))
    print("Training done")
