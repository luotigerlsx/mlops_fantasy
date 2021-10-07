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
from absl import logging

from kfp.v2.google.client import AIPlatformClient


def run_training_pipeline():
  parser = argparse.ArgumentParser()
  parser.add_argument('--project_id', type=str)
  parser.add_argument('--pipeline_region', type=str)
  parser.add_argument('--pipeline_root', type=str)
  parser.add_argument('--pipeline_job_spec_path', type=str)
  # Staging path for running custom job
  parser.add_argument('--data_pipeline_root', type=str)
  # Parameters required for data ingestion and processing
  parser.add_argument('--input_dataset_uri', type=str)
  parser.add_argument('--gcs_data_output_folder', type=str)
  parser.add_argument('--data_region', type=str)
  parser.add_argument('--gcs_result_folder', type=str)
  # Parameters required for training job
  parser.add_argument('--model_resource_name', type=str, default='')
  parser.add_argument('--endpoint_resource_name', type=str, default='')

  parser.add_argument('--machine_type', type=str, default='n1-standard-4')
  parser.add_argument('--accelerator_count', type=int, default=0)
  parser.add_argument('--accelerator_type',
                      type=str, default='ACCELERATOR_TYPE_UNSPECIFIED')
  parser.add_argument('--starting_replica_count', type=int, default=1)
  parser.add_argument('--max_replica_count', type=int, default=2)
  parser.add_argument('--pipeline_schedule',
                      type=str, default='', help='0 2 * * *')
  parser.add_argument('--pipeline_schedule_timezone',
                      type=str, default='US/Pacific')

  args, _ = parser.parse_known_args()
  logging.info(args)

  api_client = AIPlatformClient(args.project_id, args.pipeline_region)

  pipeline_params = vars(args).copy()
  pipeline_params.pop('pipeline_region', None)
  pipeline_params.pop('pipeline_root', None)
  pipeline_params.pop('pipeline_job_spec_path', None)
  pipeline_params.pop('pipeline_schedule', None)
  pipeline_params.pop('pipeline_schedule_timezone', None)

  if not args.pipeline_schedule:
    api_client.create_run_from_job_spec(
      args.pipeline_job_spec_path,
      pipeline_root=args.pipeline_root,
      parameter_values=pipeline_params,
      enable_caching=False
    )
  else:
    api_client.create_schedule_from_job_spec(
      args.pipeline_job_spec_path,
      schedule=args.pipeline_schedule,
      time_zone=args.pipeline_schedule_timezone,
      pipeline_root=args.pipeline_root,
      parameter_values=pipeline_params,
      enable_caching=False
    )


if __name__ == "__main__":
  run_training_pipeline()
