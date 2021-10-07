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

AF_REGISTRY_LOCATION=asia-southeast1
AF_REGISTRY_NAME=mlops-vertex-kit

DATA_SCHEMA='VWT:float;SWT:float;KWT:float;Entropy:float;Class:int'
TEST_INSTANCE='[{"VWT": 3.6216, "SWT": 8.6661, "KWT": -2.8073, "Entropy": -0.44699, "Class": "0"}]'

cd "$( dirname "${BASH_SOURCE[0]}" )" || exit
DIR="$( pwd )"
SRC_DIR=${DIR}"/../"
export PYTHONPATH=${PYTHONPATH}:${SRC_DIR}
echo "PYTHONPATH="${PYTHONPATH}

python -m pipelines.training_pipeline_runner \
  --project_id "$PROJECT_ID" \
  --pipeline_region asia-southeast1 \
  --pipeline_root gs://vertex_pipeline_demo_root/pipeline_root \
  --pipeline_job_spec_path ./pipeline_spec/training_pipeline_job.json \
  --data_pipeline_root gs://vertex_pipeline_demo_root/compute_root \
  --input_dataset_uri bq://"$PROJECT_ID".vertex_pipeline_demo.banknote_authentication \
  --training_data_schema ${DATA_SCHEMA} \
  --data_region asia-southeast1 \
  --gcs_data_output_folder gs://vertex_pipeline_demo_root/datasets/training \
  --training_container_image_uri ${AF_REGISTRY_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${AF_REGISTRY_NAME}/training:latest \
  --serving_container_image_uri ${AF_REGISTRY_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${AF_REGISTRY_NAME}/serving:latest \
  --custom_job_service_account 297370817971-compute@developer.gserviceaccount.com \
  --vpc_network "" \
  --hptune_region asia-east1 \
  --metrics_name au_prc \
  --metrics_threshold 0.4 \
  --endpoint_machine_type n1-standard-4 \
  --endpoint_min_replica_count 1 \
  --endpoint_max_replica_count 2 \
  --endpoint_test_instances ${TEST_INSTANCE} \
  --monitoring_user_emails luoshixin@google.com \
  --monitoring_log_sample_rate 0.8 \
  --monitor_interval 3600 \
  --monitoring_default_threshold 0.3 \
  --monitoring_custom_skew_thresholds VWT:.5,SWT:.2,KWT:.7,Entropy:.4 \
  --monitoring_custom_drift_thresholds VWT:.5,SWT:.2,KWT:.7,Entropy:.4 \
  --enable_model_monitoring True
