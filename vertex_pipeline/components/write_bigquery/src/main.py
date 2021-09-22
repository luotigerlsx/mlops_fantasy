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

import logging
from typing import Tuple

from google.cloud import bigquery
from kfp.components.executor import Executor
from kfp.dsl.io_types import Dataset, Input


def _bq_uri_to_fields(uri: str) -> Tuple[str, str, str]:
    uri = uri[5:]
    project, dataset, table = uri.split('.')
    return project, dataset, table


def load_csv_to_bigquery(
        project_id: str,
        data_region: str,
        output_dataset_uri: str,
        input_dataset: Input[Dataset]
):
    """ Component to write a CSV dataset to a BQ table.

    Args:
        project_id: The project ID.
        data_region: The region for the BQ loading job.
        output_dataset_uri: The BQ table URI to load the results.
        input_dataset: The input artifact of the prediction results dataset.
    """
    logging.getLogger().setLevel(logging.INFO)

    # Parse the source table
    logging.info(f'Input dataset URI: {input_dataset.uri}')

    output_project, output_dataset, output_table = _bq_uri_to_fields(output_dataset_uri)
    table_id = f'{output_project}.{output_dataset}.{output_table}'

    # Construct a BigQuery client object.
    client = bigquery.Client(project=project_id, location=data_region)

    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField('factset_entity_id', 'STRING'),
            bigquery.SchemaField('fsym_id', 'STRING'),
            bigquery.SchemaField('fsym_id_s', 'STRING'),
            bigquery.SchemaField('scoring_date', 'DATE'),
            bigquery.SchemaField('proper_name', 'STRING'),
            bigquery.SchemaField('prediction', 'FLOAT'),
            bigquery.SchemaField('pipeline_run_id', 'STRING'),
            bigquery.SchemaField('pipeline_ingestion_time', 'TIMESTAMP')
        ],
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )

    load_job = client.load_table_from_uri(
        input_dataset.uri, table_id, job_config=job_config
    )  # Make an API request.

    load_job.result()  # Waits for the job to complete.
    logging.info('Data import is completed')


def executor_main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--executor_input', type=str)
    parser.add_argument('--function_to_execute', type=str)

    args, _ = parser.parse_known_args()
    executor_input = json.loads(args.executor_input)
    function_to_execute = globals()[args.function_to_execute]

    executor = Executor(executor_input=executor_input,
                        function_to_execute=function_to_execute)

    executor.execute()


if __name__ == '__main__':
    executor_main()
