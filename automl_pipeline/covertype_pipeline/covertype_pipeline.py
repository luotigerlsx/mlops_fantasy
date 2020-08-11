import os
import time
from typing import NamedTuple

import kfp
from google.cloud import bigquery
from jinja2 import Template
from kfp.components import func_to_container_op

BASE_IMAGE = os.getenv('BASE_IMAGE')
COMPONENT_URL_SEARCH_PREFIX = os.getenv('COMPONENT_URL_SEARCH_PREFIX')

FEATURE_TABLE_ID = 'model_features_{}'
AUTOML_DATASET_NAME = 'cover_dataset_{}'
AUTOML_MODEL_NAME = 'cover_model_{}'

component_store = kfp.components.ComponentStore(
    local_search_paths=None, url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX]
)

bigquery_query_op = component_store.load_component('bigquery/query')
automl_create_dataset_op = component_store.load_component('automl/create_dataset_for_tables')
automl_import_data_from_bq_op = component_store.load_component('automl/import_data_from_bigquery')
automl_create_model_op = component_store.load_component('automl/create_model_for_tables')
automl_split_dataset_table_column_names_op = component_store.load_component('automl/split_dataset_table_column_names')


def prepare_query(project_id, dataset_id):
    query_template = """
        SELECT *
        FROM `{{ project_id }}.{{ dataset_id }}.covertype`
        WHERE Soil_Type IN (
          SELECT Soil_Type FROM `{{ project_id }}.{{ dataset_id }}.covertype`
          GROUP BY 1
          HAVING COUNT(*) > 500
        )
        """

    return Template(query_template).render(
        project_id=project_id, dataset_id=dataset_id)


def create_bq_job_config():
    bq_job_config = bigquery.QueryJobConfig()
    bq_job_config.create_disposition = bigquery.job.CreateDisposition.CREATE_IF_NEEDED
    bq_job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

    return bq_job_config.to_api_repr()


def retrieve_classification_metrics(project_id: str,
                                    region: str,
                                    model_path: str,
                                    metric_name: str) -> NamedTuple('Outputs', [('metric_value', float)]):
    from google.cloud import automl_v1beta1 as automl

    client = automl.TablesClient(project=project_id, region=region)
    evaluation = None
    for ev in client.list_model_evaluations(model_name=model_path):
        if ev.display_name == '':
            evaluation = ev
            break

    return (getattr(evaluation.classification_evaluation_metrics, metric_name),)


def automl_deploy_model(project_id: str,
                        region: str,
                        model_path: str):
    import logging
    from google.cloud import automl_v1beta1 as automl
    from google.cloud.automl_v1beta1 import enums

    logging.basicConfig(level=logging.INFO)
    client = automl.TablesClient(project=project_id, region=region)

    model = client.get_model(model_name=model_path)
    if model.deployment_state != enums.Model.DeploymentState.DEPLOYED:
        logging.info("Starting model deployment: {}".format(model_path))
        response = client.deploy_model(model_name=model_path)
        response.result()  # Wait for operation to complete
        logging.info("Deployment completed")
    else:
        logging.info("Model already deployed")


retrieve_classification_metrics_op = func_to_container_op(retrieve_classification_metrics, base_image=BASE_IMAGE)
automl_deploy_model_op = func_to_container_op(automl_deploy_model, base_image=BASE_IMAGE)


@kfp.dsl.pipeline(
    name='BigQuery & AutoML Cover Pipeline',
    description='The pipeline to train an AutoML Tables model based on BigQuery dataset',
)
def bq_automl_pipeline(project_id,
                       region: str,
                       dataset_id: str,
                       dataset_location: str = "US",
                       optimization_objective: str = "MINIMIZE_LOG_LOSS",
                       evaluation_metrics: str = "log_loss",
                       deployment_threshold: float = 0.1,
                       train_budget: int = 1000):
    current_milliseconds = int(time.time() * 1000.0)
    output_table_id = FEATURE_TABLE_ID.format(current_milliseconds)

    prepare_features = bigquery_query_op(
        query=prepare_query(project_id, dataset_id),
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=output_table_id,
        dataset_location=dataset_location,
        job_config=create_bq_job_config()
    )

    # Create AutoML tables dataset
    create_dataset = automl_create_dataset_op(
        gcp_project_id=project_id,
        gcp_region=region,
        display_name=AUTOML_DATASET_NAME.format(current_milliseconds)
    )
    create_dataset.after(prepare_features)

    # Import the features from BigQuery to AutoML Dataset
    import_data = automl_import_data_from_bq_op(
        dataset_path=create_dataset.outputs['dataset_path'],
        input_uri='bq://{}.{}.{}'.format(project_id, dataset_id, output_table_id)
    )

    # Set the target column label
    split_column_specs = automl_split_dataset_table_column_names_op(
        dataset_path=import_data.outputs['dataset_path'],
        table_index=0,
        target_column_name='Cover_Type'
    )

    # Create a model
    create_model = automl_create_model_op(
        gcp_project_id=project_id,
        gcp_region=region,
        display_name=AUTOML_MODEL_NAME.format(current_milliseconds),
        dataset_id=create_dataset.outputs['dataset_id'],
        target_column_path=split_column_specs.outputs['target_column_path'],
        input_feature_column_paths=split_column_specs.outputs['feature_column_paths'],
        optimization_objective=optimization_objective,
        train_budget_milli_node_hours=train_budget
    )

    # Retrieve the evaluation metric from the model evaluations
    # test_model_path='projects/66820330676/locations/us-central1/models/TBL237954107460026368'
    retrieve_metrics = retrieve_classification_metrics_op(
        project_id=project_id,
        region=region,
        model_path=create_model.outputs['model_path'],
        metric_name=evaluation_metrics)

    # Deploy the model if the primary metric is better than threshold
    with kfp.dsl.Condition(retrieve_metrics.outputs['metric_value'] < deployment_threshold):
        deploy_model = automl_deploy_model_op(project_id, region, create_model.outputs['model_path'])
