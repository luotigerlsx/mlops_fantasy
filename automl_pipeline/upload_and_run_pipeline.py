import logging
import time

import fire
import kfp

logging.basicConfig(level=logging.INFO)


def upload_and_run_pipeline(endpoint, pipeline_name, pipeline_version, pipeline_package,
                            experiment_name, run=False, params=None):
    client = kfp.Client(endpoint)

    logging.info('Getting existing pipeline by name')
    try:
        pipeline_id = client.get_pipeline_id(pipeline_name)
    except TypeError:
        logging.info('Pipeline does not exist, upload pipeline package')
        response = client.upload_pipeline(pipeline_package_path=pipeline_package, pipeline_name=pipeline_name)
        pipeline_id = response.id

    logging.info('Pipeline ID: {}'.format(pipeline_id))
    logging.info('Upload pipeline version: {}'.format(pipeline_version))
    version = client.upload_pipeline_version(pipeline_package, pipeline_version, pipeline_id=pipeline_id)
    logging.info('Pipeline version with ID {} is created'.format(version.id))

    if run:
        logging.info('Run pipeline in experiment {}'.format(experiment_name))
        experiment = client.get_experiment(experiment_name=experiment_name)

        job_name = '{}_{}'.format(pipeline_name, int(time.time() * 1000.0))
        client.run_pipeline(experiment_id=experiment.id, job_name=job_name, pipeline_id=pipeline_id,
                            version_id=version.id, params=params)


if __name__ == '__main__':
    fire.Fire(upload_and_run_pipeline)
