"""MLCompass state management library."""

import dataclasses
import os
from typing import TYPE_CHECKING, Any, Optional, cast, Dict, List
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
class CommitInfo:
  repo: str
  commit: str
  branch: str


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class BenchmarkData:
  """Subset of MLCompass benchmark data."""
  test_name: str
  xlml_node_id: str
  succeeded: Optional[bool] = None
  error_message: Optional[str] = None
  metrics: Optional[Dict[str, float]] = None
  commit_map: Optional[Dict[str, CommitInfo]] = None
  link_map: Optional[Dict[str, str]] = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class MLCompassState:
  """State for running XLML benchmarks."""
  state_uuid: Optional[str] = None
  mlcompass_tracking_id: Optional[str] = None
  execution_mode: Optional[str] = None
  environment_name: Optional[str] = None
  dag_name: Optional[str] = None
  dag_run_id: Optional[str] = None
  airflow_gcp_project: Optional[str] = None
  airflow_gcp_location: Optional[str] = None
  workdir_bucket: Optional[str] = None
  workdir_path: Optional[str] = None
  benchmarks: List[BenchmarkData] = dataclasses.field(default_factory=list)


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
    state.airflow_gcp_project = state.airflow_gcp_project or os.getenv('GCP_PROJECT')
    state.airflow_gcp_location = state.airflow_gcp_location or os.getenv('COMPOSER_LOCATION')
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


class ScheduleOperator(baseoperator.BaseOperator, skipmixin.SkipMixin):
  """An operator to schedule MLCompass benchmarks."""

  def __init__(self, *, node_map: dict[str, list[taskmixin.DAGNode]], **kwargs) -> None:
    super().__init__(**kwargs)
    self.node_map = node_map

  def execute(self, context: Context) -> None:
    state = load_state(context)
    if not state:
      self.log.info('No MLCompass state found; scheduling all benchmarks.')
      return []
    self.log.info(f'Loaded MLCompass state: {state.to_json(indent=2)}')
    dag_run = context['dag_run']
    skip_tasks = []
    node_id_to_benchmarks = {benchmark.xlml_node_id:benchmark for benchmark in state.benchmarks}
    for key, nodes in self.node_map.items():
      if key in node_id_to_benchmarks:
        self.log.info(f'Scheduling benchmark: {key}')
        node_id_to_benchmarks[key].succeeded = False
        node_id_to_benchmarks[key].error_message = 'Scheduled, but not post processed in XLML.'
      else:
        self.log.info(f'Skipping benchmark: {key}')
        skip_tasks.extend(nodes)
    for benchmark in state.benchmarks:
      if benchmark.succeeded is None:
        self.log.info(f'No task found for Benchmark {benchmark.xlml_node_id}.')
        benchmark.succeeded = False
        benchmark.error_message = f'No corresponding task found with name {benchmark.xlml_node_id} in XLML.'
    save_state(state)
    self.skip(
        dag_run=dag_run,
        execution_date=cast(DateTime, dag_run.execution_date),
        tasks=skip_tasks,
        map_index=context["ti"].map_index,
    )

class PostProcessOperator(baseoperator.BaseOperator):
  """An operator to perform post-processing after MLCompass benchmarks."""

  def __init__(self, *, node_map: dict[str, list[taskmixin.DAGNode]], **kwargs) -> None:
    super().__init__(**kwargs)
    self.node_map = node_map

  def execute(self, context: Context) -> None:
    state = load_state(context)
    if not state:
      self.log.info('No MLCompass state found; Skipping post-processing.')
      return
    # node_id_to_benchmarks = {benchmark.xlml_node_id:benchmark for benchmark in state.benchmarks}
    # for key, nodes in self.node_map.items():
    #   if key in node_id_to_benchmarks:
    #     benchmark = node_id_to_benchmarks[key]
    #     all_succeeded = True

    print('Completed processing for benchmarks')


def get_all_tasks(node: taskmixin.DAGNode) -> list[taskmixin.DAGNode]:
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


class MLCompass:
  def __init__(self) -> None:
      self.node_map = {}
      self.schedule = ScheduleOperator(task_id='mlcompass_schedule', node_map=self.node_map)
      self.post_process = PostProcessOperator(task_id='mlcompass_post_process', node_map=self.node_map, trigger_rule='all_done')
      self.schedule >> self.post_process

  def register(self, node: taskmixin.DAGNode) -> None:
      tasks = get_all_tasks(node)
      self.node_map[node.node_id] = tasks
      self.schedule >> node >> self.post_process

