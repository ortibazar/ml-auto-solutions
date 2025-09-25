"""MLCompass state management library."""

import dataclasses
import os
from typing import TYPE_CHECKING, Any, Optional, cast
from pendulum.datetime import DateTime
from airflow.decorators import task
from airflow.models import taskmixin, skipmixin, baseoperator, abstractoperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.utils.context import Context
from airflow.utils.task_group import TaskGroup
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


class ScheduleOperator(baseoperator.BaseOperator, skipmixin.SkipMixin):
  """An operator to schedule MLCompass benchmarks."""

  def __init__(self, *, node_map: dict[str, list[taskmixin.DAGNode]], **kwargs) -> None:
    super().__init__(**kwargs)
    self.node_map = node_map

  def execute(self, context: Context) -> None:
    dag_run = context['dag_run']
    tasks = []
    for key, nodes in self.node_map.items():
      if key not in ('maxtext-gpt3_175b-stable-1-2xv6e-256'):
        tasks.extend(nodes)
    self.skip(
        dag_run=dag_run,
        execution_date=cast(DateTime, dag_run.execution_date),
        tasks=tasks,
        map_index=context["ti"].map_index,
    )

@task(task_id='mlcompass_post_process', trigger_rule='all_done', retries=0)
def mlcompass_post_process(node_map: dict[str, list[taskmixin.DAGNode]]) -> None:
  print(f'Completed processing for benchmarks: {node_map}')

def get_descendants(node: taskmixin.DAGNode) -> list[taskmixin.DAGNode]:
  """Get all descendant nodes of a given DAGNode."""
  if isinstance(node, abstractoperator.AbstractOperator):
    return [node]
  if not isinstance(node, TaskGroup):
    raise ValueError(
        f"Input node is not a TaskGroup or an AbstractOperator: {type(node)}"
    )
  result = []
  groups_to_visit = [node]
  while groups_to_visit:
      visiting = groups_to_visit.pop(0)
      for child in visiting.children.values():
          if isinstance(child, abstractoperator.AbstractOperator):
              result.append(child)
          elif isinstance(child, TaskGroup):
              groups_to_visit.append(child)
          else:
              raise ValueError(
                  f"Encountered a DAGNode that is not a TaskGroup or an AbstractOperator: {type(child)}"
              )
  return result

def register(nodes: list[taskmixin.DAGNode]) -> None:
  node_map = {node.node_id:get_descendants(node) for node in nodes}
  schedule = ScheduleOperator(task_id='mlcompass_schedule', node_map=node_map)
  post_process = mlcompass_post_process(node_map)
  schedule >> post_process
  for node in nodes:
    schedule >> node >> post_process

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
