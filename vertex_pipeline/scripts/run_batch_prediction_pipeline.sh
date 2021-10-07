#!/bin/bash

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PROJECT_ID=$(gcloud config get-value project)

ENDPOINT='projects/297370817971/locations/asia-southeast1/endpoints/8843521555783745536'


cd "$( dirname "${BASH_SOURCE[0]}" )" || exit
DIR="$( pwd )"
SRC_DIR=${DIR}"/../"
export PYTHONPATH=${PYTHONPATH}:${SRC_DIR}
echo "PYTHONPATH="${PYTHONPATH}

python -m pipelines.batch_prediction_pipeline_runner \
  --project_id "$PROJECT_ID" \
  --pipeline_region asia-southeast1 \
  --pipeline_root gs://vertex_pipeline_demo_root/pipeline_root \
  --pipeline_job_spec_path ./pipeline_spec/batch_prediction_pipeline_job.json \
  --data_pipeline_root gs://vertex_pipeline_demo_root/compute_root \
  --input_dataset_uri bq://"$PROJECT_ID".vertex_pipeline_demo.banknote_authentication_features \
  --data_region asia-southeast1 \
  --gcs_data_output_folder gs://vertex_pipeline_demo_root/datasets/prediction \
  --gcs_result_folder gs://vertex_pipeline_demo_root/prediction \
  --endpoint_resource_name $ENDPOINT\
  --machine_type n1-standard-8 \
  --accelerator_count 0 \
  --accelerator_type ACCELERATOR_TYPE_UNSPECIFIED \
  --starting_replica_count 1 \
  --max_replica_count 2
