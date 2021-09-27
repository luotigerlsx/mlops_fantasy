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
import os
from datetime import datetime

from google.cloud import aiplatform
from kfp.components.executor import Executor
from kfp.dsl.io_types import Artifact, Dataset, Input, Output

# Vertex AI artifact resource prefix
VERTEX_AI_RESOURCE_PREFIX = 'aiplatform://v1/'

# TODO: to change and use batch-prediction operation with serving container
def batch_predict(
    project_id: str,
    data_region: str,
    data_pipeline_root: str,
    pipeline_run_id: str,
    custom_container_image_uri: str,
    custom_job_service_account: str,
    scoring_id_columns: str,
    gcs_scoring_result_folder: str,
    input_dataset: Input[Dataset],
    input_endpoint: Input[Artifact],
    scoring_result: Output[Dataset],
    vpc_network: str = None,
):
  """ Perform batch prediction.

  Args:
      project_id: The project ID.
      data_region: The region for batch prediction custom job.
      data_pipeline_root: The staging location for custom job.
      pipeline_run_id: The pipeline run ID for logging purpose.
      custom_container_image_uri: The custom job container image URI.
      custom_job_service_account: The service account to run the custom job.
      scoring_id_columns: The ID columns used to key each prediction instance.
      gcs_scoring_result_folder: The GCS folder to store the scoring
        results in CSV format.
      input_dataset: The input artifact of scoring dataset.
      input_endpoint: The input artifact of prediction endpoint.
      scoring_result: The output artifact of scoring results.
      vpc_network: The VPC network to peer with (optional).
  """
  logging.getLogger().setLevel(logging.INFO)

  logging.info(f'input dataset URI: {input_dataset.uri}')
  logging.info(f'input endpoint URI: {input_endpoint.uri}')

  # Avoid writing empty pipeline_run_id in output dataset without error
  if not pipeline_run_id:
    raise RuntimeError('The argument pipeline_run_id is empty')

  scoring_result_uri = os.path.join(
    gcs_scoring_result_folder,
    f'predictions-{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')

  # Call Vertex AI custom job in another region
  aiplatform.init(
    project=project_id,
    location=data_region,
    staging_bucket=data_pipeline_root)

  worker_pool_specs = [
    {
      "machine_spec": {
        "machine_type": "n1-standard-4",
      },
      "replica_count": 1,
      "container_spec": {
        "image_uri": custom_container_image_uri,
        "command": [],
        "args": [
          '--project_id', project_id,
          '--region', data_region,
          '--pipeline_run_id', pipeline_run_id,
          '--scoring_data_uri', input_dataset.uri,
          '--scoring_result_uri', scoring_result_uri,
          '--scoring_id_columns', scoring_id_columns,
          '--endpoint_resource_name',
          input_endpoint.uri[len(VERTEX_AI_RESOURCE_PREFIX):]
        ]
      }
    }
  ]

  job = aiplatform.CustomJob(
    display_name='batch-prediction',
    project=project_id,
    location=data_region,
    worker_pool_specs=worker_pool_specs
  )

  job.run(
    service_account=custom_job_service_account,
    network=vpc_network
  )
  logging.info(f'Custom job {job.resource_name} is completed.')

  scoring_result.uri = scoring_result_uri


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
