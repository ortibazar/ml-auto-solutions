# Copyright 2024 Google LLC
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

"""
A DAG to run perf tests for MaxText model configs on v5e.
"""
import datetime
from airflow import models
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, Project, ClusterName, DockerImage
from dags.multipod.configs import maxtext_sweep_gke_config
from dags.multipod.configs.common import SetupMode
from xlml.apis import metric_config

# Run once a day at 4 am UTC (8 pm PST / 9 pm PDT)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_v5e_configs_perf",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable", "nightly"],
    start_date=datetime.datetime(2024, 2, 19),
    catchup=False,
) as dag:
  # MaxText set up
  quantization_sweep = {"M_QUANTIZATION": ["", "int8"]}
  model_configs = [
      "16b",
      "32b",
      "64b",
      "128b",
      "gpt3_175b",
      "llama2_7b",
      "llama2_13b",
      "llama2_70b",
  ]
  docker_images = [
      (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_STABLE),
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
  ]
  base_output_directory = "gs://runner-maxtext-logs"

  for mode, image in docker_images:
    for model in model_configs:
      base_run_model_cmds = [
          f"bash MaxText/configs/v5e/{model}.sh OUTPUT_PATH={base_output_directory} DATASET_PATH=gs://max-datasets-rogue PLATFORM=gke",
      ]
      maxtext_sweep_gke_test = maxtext_sweep_gke_config.get_maxtext_sweep_gke_config(
          test_owner=test_owner.RAYMOND_Z,
          project_name=Project.TPU_PROD_ENV_MULTIPOD.value,
          dataset_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
          composer_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
          dataset_name=metric_config.DatasetOption.XLML_DATASET,
          cluster_name=ClusterName.V5E_256_US_WEST_4_MULTISLICE_CLUSTER.value,
          tpu_zone=Zone.US_WEST4_B.value,
          time_out_in_min=360,
          base_output_directory=base_output_directory,
          tpu_version=TpuVersion.V5E,
          tpu_cores=256,
          num_slices=[1, 2],
          docker_image=image.value,
          run_name_prefix=f"maxtext-{model}-{mode.value}",
          base_run_model_cmds=base_run_model_cmds,
          sweep_params=quantization_sweep,
      )

      num_tests = len(maxtext_sweep_gke_test)
      for i in range(num_tests // 2):
        (
            maxtext_sweep_gke_test[i].run_with_run_name_generation()
            >> maxtext_sweep_gke_test[
                num_tests - 1 - i
            ].run_with_run_name_generation()
        )
