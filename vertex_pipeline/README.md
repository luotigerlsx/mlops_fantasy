<!--
Copyright 2021 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
-->

# Vertex AI Pipeline

This document describes the [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines)
artifacts contained in this repository in terms of pipelines, components and scripts, as well as the samples built to
illustrate their usage.

## Repository Structure

The repository contains the following:

```
.
├── scripts       : cloud build configuration and build scripts
├── components    : vertex pipeline components
├── images        : vertex pipeline images
├── pipelines     : vertex pipeline definitions 
├── configs       : vertex pipeline definitions 
└── notebooks     : notebooks used in workshop or experimentation

```

## Building and Running the Training / Prediction Pipelines

The end-to-end process of creating and running the training pipeline contains the following steps:

1. Create the components required to build and run the pipeline
2. Prepare the configuration files of the various steps of the pipeline
3. Build the pipeline
4. Run the pipeline

### Building Components

The components and supporting images can be built using the provided Cloud Build configuration files.

To build the supporting images, run the following in the source repo root directory:

```bash
sh build_images_cb.sh
```

To build the pipeline components, and the pipeline job specifications, run the following in the source repo root
directory:

```bash
sh build_components_cb.sh
```

To build the pipeline job specifications, run the following in the source repo root
directory:

```bash
sh build_pipeline_cb.sh
```

### Configuration and Execution

#### Manual Trigger

For manual triggering, the following two scripts should be updated:

* `scripts/run_training_pipeline.sh`
* `scripts/run_batch_prediction_pipeline.sh`