#!/bin/bash

pushd kfp_image
KFP_IMAGE_URI="gcr.io/${PROJECT_ID}/kfp-cli"
gcloud builds submit --timeout 15m --tag ${KFP_IMAGE_URI} .
popd
