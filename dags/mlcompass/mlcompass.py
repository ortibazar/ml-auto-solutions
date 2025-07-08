"""MLCompass state management library."""

import dataclasses
import os
from typing import Any, Optional
from airflow.providers.google.cloud.hooks.gcs import GCSHook
import dataclasses_json

STATE_BUCKET_NAME = 'mlcompass-jax-artifacts'


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class MLCompassState:
  """State for running XLML benchmarks."""

  state_uuid: Optional[str] = None
  mlcompass_tracking_id: Optional[str] = None
  benchmark_name: Optional[str] = None
  execution_mode: Optional[str] = None
  environment_name: Optional[str] = None
  dag_name: Optional[str] = None
  dag_run_id: Optional[str] = None
  airflow_gcp_project: Optional[str] = None
  airflow_gcp_location: Optional[str] = None
  workdir_bucket: Optional[str] = None
  workdir_path: Optional[str] = None
  metrics: Optional[dict[str, float]] = None


def get_state_uuid(context: Optional[dict[str, Any]]) -> Optional[str]:
  params = context.get('params')
  if params:
    return params.get('mlcompass_state_uuid')
  return None


def get_state_object_name(state_uuid: str) -> str:
  return f'xlml/{state_uuid}/mlcompass_state.json'


def load_state(context: Optional[dict[str, Any]]) -> Optional[MLCompassState]:
  state_uuid = get_state_uuid(context)
  if state_uuid:
    gcs_hook = GCSHook()
    content = gcs_hook.download(
        bucket_name=STATE_BUCKET_NAME,
        object_name=get_state_object_name(state_uuid),
    )
    state = MLCompassState.from_json(content)
    print(f'Loaded MLCompass state: {content}')
    state.dag_run_id = context['run_id']
    state.airflow_gcp_project = os.getenv('GCP_PROJECT')
    state.airflow_gcp_location = os.getenv('COMPOSER_LOCATION')
    return state
  return None


def save_state(state: MLCompassState) -> None:
  content = state.to_json(indent=2)
  gcs_hook = GCSHook()
  gcs_hook.upload(
      bucket_name=STATE_BUCKET_NAME,
      object_name=get_state_object_name(state.state_uuid),
      data=content,
      mime_type='application/json',
  )
  print(f'Saved MLCompass state: {content}')
