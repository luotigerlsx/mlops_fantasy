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
from typing import NamedTuple

from kfp.components.executor import Executor
from kfp.dsl.io_types import Input, Metrics

logging.getLogger().setLevel(logging.INFO)

supported_metrics_names = ['au_roc', 'au_prc', 'accuracy', 'precision', 'recall']


def check_metrics(
        metrics_name: str,
        metrics_threshold: float,
        basic_metrics: Input[Metrics]
) -> NamedTuple(
    'Outputs',
    [
        ('is_better_metrics', str)
    ]
):
    """ Check the model metrics to see if it should be deployed.

    Args:
        metrics_name: The model metrics to check.
        metrics_threshold: The metrics value.
        basic_metrics: The input artifact of the simple model metrics.

    Returns:
        A NamedTuple containing the result. If the model performance is larger than the threshold, it returns 'True'.
        Otherwise, it returns 'False'.
    """

    if metrics_name not in supported_metrics_names:
        raise RuntimeError(f'Metrics name {metrics_name} is not supported')

    metrics_value = basic_metrics.metadata[metrics_name]
    is_better_metrics = 'True' if metrics_value >= metrics_threshold else 'False'

    return (is_better_metrics,)


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
