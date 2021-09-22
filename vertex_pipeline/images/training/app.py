# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import json
import logging
import os
import re
from typing import Dict, Tuple

import gcsfs
import lightgbm as lgb
import pandas as pd
import yaml
from google.cloud import aiplatform_v1beta1
from google.cloud.aiplatform_v1beta1 import Study, Trial
from lightgbm import Booster
from numpy.random import RandomState
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)

################################################################################
# Model serialization code
################################################################################

MODEL_FILENAME = 'model.txt'
FEATURE_IMPORTANCE_FILENAME = 'feature_importance.csv'
INSTANCE_SCHEMA_FILENAME = 'instance_schema.yaml'
PROB_THRESHOLD = 0.5


def _save_model(model: Booster, model_store: str):
    model.save_model(MODEL_FILENAME)
    fs = gcsfs.GCSFileSystem()
    fs.put_file(MODEL_FILENAME, os.path.join(model_store, MODEL_FILENAME))


def _save_feature_importance(model: Booster, model_store: str):
    file_path = os.path.join(model_store, FEATURE_IMPORTANCE_FILENAME)
    imp = pd.DataFrame(
        {
            'feature': model.feature_name(),
            'importance': model.feature_importance()
        }
    )
    imp.to_csv(file_path, index=False)


def _save_analysis_schema(df: pd.DataFrame, model_store: str):
    # create feature schema
    properties = {}
    required = df.columns.tolist()

    for feature in required:
        if feature in ['sic', 'industry', 'sector']:
            properties[feature] = {'type': 'string', 'nullable': True}
        else:
            properties[feature] = {'type': 'number', 'nullable': True}

    spec = {
        'type': 'object',
        'properties': properties,
        'required': required
    }

    fs = gcsfs.GCSFileSystem()
    with fs.open(os.path.join(model_store, INSTANCE_SCHEMA_FILENAME), 'w') as file:
        yaml.dump(spec, file)


def _save_metrics(metrics: dict, output_path: str):
    fs = gcsfs.GCSFileSystem()
    with fs.open(output_path, 'wt') as eval_file:
        eval_file.write(json.dumps(metrics))


################################################################################
# Model training
################################################################################


def train(args: argparse.Namespace):
    if 'AIP_MODEL_DIR' not in os.environ:
        raise KeyError(
            'The `AIP_MODEL_DIR` environment variable has not been' +
            'set. See https://cloud.google.com/ai-platform-unified/docs/tutorials/image-recognition-custom/training'
        )
    output_model_directory = os.environ['AIP_MODEL_DIR']

    logging.info(f'AIP_MODEL_DIR: {output_model_directory}')
    logging.info(f'training_data_uri: {args.training_data_uri}')
    logging.info(f'metrics_output_uri: {args.metrics_output_uri}')

    # prepare the data
    x_train, y_train = _load_csv_dataset(args.training_data_uri, args.target_label)

    # validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=RandomState(42))

    # test data
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=RandomState(42))

    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature="auto")
    lgb_val = lgb.Dataset(x_val, y_val, categorical_feature="auto")

    # Conduct Vizier trials a.k.a. hyperparameter tuning prior to main training activity
    best_num_leaves, best_max_depth = _conduct_vizier_trials(args, lgb_train, lgb_val, x_test, y_test)
    logging.info(f'Vizier best_num_leaves: {best_num_leaves}')
    logging.info(f'Vizier best_max_depth: {best_max_depth}')

    # Train model with best hyperparameters from the Vizier study
    logging.info(f'Training with best hyperparameters from Vizier')
    model = _training_execution(
        lgb_train, lgb_val, int(args.num_boost_round), best_num_leaves, best_max_depth, int(args.min_data_in_leaf)
    )

    # save the generated model
    _save_model(model, output_model_directory)
    _save_feature_importance(model, output_model_directory)
    _save_analysis_schema(x_train, output_model_directory)

    # save eval metrics
    metrics = _evaluate_model(model, x_test, y_test)
    _save_metrics(metrics, args.metrics_output_uri)


def _training_execution(
        lgb_train: lgb.Dataset,
        lgb_val: lgb.Dataset,
        num_boost_round: int,
        num_leaves: int,
        max_depth: int,
        min_data_in_leaf: int
) -> lgb.Booster:
    # train the model
    params = {
        'objective': 'binary',
        'is_unbalance': True,
        'boosting_type': 'gbdt',
        'metric': ['auc'],
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'min_data_in_leaf': min_data_in_leaf
    }

    evals_result = {}  # to record eval results
    model = lgb.train(params=params,
                      num_boost_round=num_boost_round,
                      train_set=lgb_train,
                      valid_sets=[lgb_val, lgb_train],
                      valid_names=["test", "train"],
                      evals_result=evals_result,
                      verbose_eval=True)

    return model


def _split_features_label_columns(df: pd.DataFrame, target_label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[target_label]
    x = df.drop(target_label, axis=1)

    return x, y


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    # Gather feature column names.
    feature_col_ff = [col for col in df.columns if col.startswith("ff_")]
    feature_col_fp = [col for col in df.columns if re.match("(p|v|ret)_", col)]
    feature_col_own = [col for col in df.columns if col.startswith("inst_")]
    feature_col_own += [col for col in df.columns if col.startswith("insider_pos")]
    feature_col_own += [col for col in df.columns if col.startswith("fund_h")]
    feature_col_own += [col for col in df.columns if col.startswith("fund_m")]
    feature_col_own += [col for col in df.columns if col.startswith("insider_order")]
    feature_col_cat = ["sic", "industry", "sector"]  # Categorical.
    feature_cols = [
        feature_col_ff,
        feature_col_fp,
        feature_col_own,
        feature_col_cat,
    ]
    feature_cols = sum(feature_cols, [])  # Flatten.
    logging.info(f"Total number of feature columns: {len(feature_cols)}")

    # Encode categorical features.
    for col in feature_col_cat:
        df[col] = df[col].astype("category")

    return df[feature_cols]


def _load_csv_dataset(data_uri_pattern: str, target_label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fs = gcsfs.GCSFileSystem()

    all_files = fs.glob(data_uri_pattern)
    df = pd.concat((pd.read_csv('gs://' + f) for f in all_files), ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    x, y = _split_features_label_columns(df, target_label)
    x = _select_features(x)

    return x, y


def _evaluate_model(model: Booster, x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, object]:
    # get roc curve metrics, down sample to avoid hitting MLMD 64k size limit
    roc_size = int(x.shape[0] * 1 / 3)
    y_hat = model.predict(x)
    pred = (y_hat > PROB_THRESHOLD).astype(int)

    fpr, tpr, thresholds = roc_curve(
        y_true=y[:roc_size], y_score=y_hat[:roc_size], pos_label=True
    )

    # get classification metrics
    au_roc = roc_auc_score(y, y_hat)
    au_prc = average_precision_score(y, y_hat)
    classification_metrics = classification_report(y, pred, output_dict=True)
    confusion_mat = confusion_matrix(y, pred, labels=[0, 1])

    metrics = {
        'classification_report': classification_metrics,
        'confusion_matrix': confusion_mat.tolist(),
        'au_roc': au_roc,
        'au_prc': au_prc,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }

    return metrics


################################################################################
# Hyperparameter tuning using Vertex Vizier.
################################################################################


def _create_study(vizier_client: aiplatform_v1beta1.VizierServiceClient, args: argparse.Namespace) -> Study:
    study_display_name = '{}_study_{}'.format(
        args.hp_config_gcp_project_id.replace('-', ''),
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    parent = 'projects/{}/locations/{}'.format(args.hp_config_gcp_project_id, args.hp_config_gcp_region)

    param_num_leaves = {
        'parameter_id': 'num_leaves',
        'integer_value_spec': {
            'min_value': int(args.num_leaves_hp_param_min),
            'max_value': int(args.num_leaves_hp_param_max)
        }
    }

    param_max_depth = {
        'parameter_id': 'max_depth',
        'integer_value_spec': {
            'min_value': int(args.max_depth_hp_param_min),
            'max_value': int(args.max_depth_hp_param_max)
        }
    }

    # Objective Metric
    metric_auc = {
        'metric_id': 'auc',
        'goal': 'MAXIMIZE'
    }

    # Create study using define parameter and metric
    study_spec = {
        'display_name': study_display_name,
        'study_spec': {
            'parameters': [
                param_num_leaves,
                param_max_depth,
            ],
            'metrics': [metric_auc],
        }
    }

    logging.info(f'Vizier study spec {study_spec}')

    # Create study
    return vizier_client.create_study(parent=parent, study=study_spec)


def _get_trial_parameters(trial: Trial) -> Tuple[int, int]:
    num_leaves = None
    max_depth = None

    for param in trial.parameters:
        if param.parameter_id == "num_leaves":
            num_leaves = int(param.value)
        elif param.parameter_id == "max_depth":
            max_depth = int(param.value)

    return num_leaves, max_depth


def _conduct_vizier_trials(
        args: argparse.Namespace,
        lgb_train: lgb.Dataset,
        lgb_val: lgb.Dataset,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame
) -> Tuple[int, int]:
    logging.info(f'Commencing Vizier study')

    endpoint = args.hp_config_gcp_region + '-aiplatform.googleapis.com'
    # Define Vizier client
    vizier_client = aiplatform_v1beta1.VizierServiceClient(
        client_options=dict(api_endpoint=endpoint)
    )

    vizier_study = _create_study(vizier_client, args)
    vizier_study_id = vizier_study.name

    logging.info(f'Vizier study name: {vizier_study_id}')

    # Conduct training trials using Vizier generated params
    client_id = "shareholder_training_job"
    suggestion_count_per_request = int(args.hp_config_suggestions_per_request)
    max_trial_id_to_stop = int(args.hp_config_max_trials)

    trial_id = 0
    while int(trial_id) < max_trial_id_to_stop:
        suggest_response = vizier_client.suggest_trials(
            {
                "parent": vizier_study_id,
                "suggestion_count": suggestion_count_per_request,
                "client_id": client_id,
            }
        )

        for suggested_trial in suggest_response.result().trials:
            trial_id = suggested_trial.name.split("/")[-1]
            trial = vizier_client.get_trial({"name": suggested_trial.name})

            logging.info(f'Vizier trial start {trial_id}')

            if trial.state in ["COMPLETED", "INFEASIBLE"]:
                continue

            num_leaves, max_depth = _get_trial_parameters(trial)

            model = _training_execution(
                lgb_train, lgb_val, int(args.num_boost_round), num_leaves, max_depth, int(args.min_data_in_leaf)
            )

            # Get model evaluation metrics
            metrics = _evaluate_model(model, x_test, y_test)

            # Log measurements back to vizier
            vizier_client.add_trial_measurement(
                {
                    "trial_name": suggested_trial.name,
                    "measurement": {
                        'metrics': [{'metric_id': 'auc', 'value': metrics['au_roc']}]
                    },
                }
            )

            # Complete the Vizier trial
            vizier_client.complete_trial(
                {"name": suggested_trial.name, "trial_infeasible": False}
            )

            logging.info(f'Vizier trial completed {trial_id}')

    # Get the optimal trail with the best ROC AUC
    optimal_trials = vizier_client.list_optimal_trials({"parent": vizier_study_id})

    # Extract best hyperparams from best trial
    best_num_leaves, best_max_depth = _get_trial_parameters(optimal_trials.optimal_trials[0])

    return best_num_leaves, best_max_depth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_uri', type=str,
                        help='The training dataset location in GCS.')
    parser.add_argument('--target_label', type=str,
                        help='The target feature name in the dataset.')
    parser.add_argument('--metrics_output_uri', type=str,
                        help='The GCS artifact URI to write model metrics.')
    parser.add_argument('--min_data_in_leaf', dest='min_data_in_leaf',
                        default=5, type=float,
                        help='Minimum number of observations that must fall into a tree node for it to be added.')
    parser.add_argument('--num_boost_round', dest='num_boost_round',
                        default=300, type=float,
                        help='Number of boosting iterations.')
    parser.add_argument('--max_depth_hp_param_min', dest='max_depth_hp_param_min',
                        default=-1, type=float,
                        help='Max tree depth for base learners, <=0 means no limit. Min value for hyperparam param')
    parser.add_argument('--max_depth_hp_param_max', dest='max_depth_hp_param_max',
                        default=3, type=float,
                        help='Max tree depth for base learners, <=0 means no limit.  Max value for hyperparam param')
    parser.add_argument('--num_leaves_hp_param_min', dest='num_leaves_hp_param_min',
                        default=6, type=float,
                        help='Maximum tree leaves for base learners. Min value for hyperparam param.')
    parser.add_argument('--num_leaves_hp_param_max', dest='num_leaves_hp_param_max',
                        default=10, type=float,
                        help='Maximum tree leaves for base learners. Max value for hyperparam param.')
    parser.add_argument('--hp_config_max_trials', dest='hp_config_max_trials',
                        default=20, type=float,
                        help='Maximum number of hyperparam tuning trials.')
    parser.add_argument('--hp_config_suggestions_per_request', dest='hp_config_suggestions_per_request',
                        default=5, type=float,
                        help='Suggestions per vizier request')
    parser.add_argument('--hp_config_gcp_region', dest='hp_config_gcp_region',
                        default='asia-east1', type=str,
                        help='Vizier GCP Region. Data or model no passed to vizier. Simply tuning config.')
    parser.add_argument('--hp_config_gcp_project_id', dest='hp_config_gcp_project_id',
                        default='uob-ml-deployment', type=str,
                        help='GCP project id.')
    train(parser.parse_args())
