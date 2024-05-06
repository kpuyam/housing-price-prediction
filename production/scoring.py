"""Processors for the model scoring/evaluation step of the worklow."""
import os.path as op
import pandas as pd
import logging
from ta_lib.core.api import (
    load_dataset,
    load_pipeline,
    register_processor,
    save_dataset,
    DEFAULT_ARTIFACTS_PATH
)
from ta_lib.regression.api import RegressionComparison
logger = logging.getLogger(__name__)


@register_processor("model-eval", "score-model")
def score_model(context, params):
    """
    Score a pre-trained model.

    Parameters:
    - context: The context object containing information about the execution environment.
    - params: Additional parameters or configuration settings for scoring the model.

    Returns:
    - None
    
    Output dataset: "score/housing/output"
    artifacts_folder: Path to folder to store artifacts

    Description:
    - Load test datasets from the specified path
    - Load the feature pipeline and training pipelines from the artifacts folder.
    - Transform the test features dataset using the loaded pipelines to generate predictions.
    - Save the transformed test features dataset to the specified output dataset path.
    - Export performance metrics and comparison report for the model predictions.
    """
    output_ds = "score/housing/output"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load test datasets
    test_X = load_dataset(context, "test/housing/features")
    test_y = load_dataset(context, "test/housing/target")

    # load the feature pipeline and training pipelines
    full_pipeline = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    lin_reg_ppln = load_pipeline(op.join(artifacts_folder, "lin_reg_pipeline.joblib"))
    dtree_reg_ppln = load_pipeline(op.join(artifacts_folder, "dtree_reg_pipeline.joblib"))
    rand_search = load_pipeline(op.join(artifacts_folder, "rand_search.joblib"))
    grid_search = load_pipeline(op.join(artifacts_folder, "grid_search.joblib"))

    # transform the test dataset
    test_X_pred = pd.DataFrame()
    test_X_pred["lin_y_pred"] = lin_reg_ppln.predict(test_X)
    test_X_pred['dtree_pred'] = dtree_reg_ppln.predict(test_X)
    test_X_pred['rand_search'] = rand_search.predict(test_X)
    test_X_pred['grid_search'] = grid_search.predict(test_X)
    print(test_X_pred)
    test_X = test_X.join(test_X_pred)
    print(test_X)
    save_dataset(context, test_X, output_ds)
    
    model_pipelines = ['LinReg', 'Dtree', 'RandomizedSearchCV', 'GridSearch']
    predictions = [
        test_X["lin_y_pred"],
        test_X['dtree_pred'],
        test_X['rand_search'],
        test_X['grid_search']
    ]
    predictions_dict = dict(zip(model_pipelines, predictions))
    model_comparison_report_1 = RegressionComparison(
        y=test_y, yhats=predictions_dict
    )

    report_metrics = model_comparison_report_1.perf_metrics()
    metrics = model_comparison_report_1.get_report(file_path="production/reports/ta_reg_comparison")
    print(model_comparison_report_1.performance_metrics)
