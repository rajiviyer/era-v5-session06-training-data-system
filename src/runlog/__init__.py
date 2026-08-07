"""Structured run.log: the run's ordered event stream (SCOPE.md §9.1, P11-T01)."""

from .reader import event_type_counts, events_of_type, load_run_log, missing_event_types
from .types import EVENT_TYPES, RUN_LOG_FILENAME, RunLogError, RunLogEvent
from .writer import RunLogWriter

__all__ = [
    "EVENT_TYPES",
    "RUN_LOG_FILENAME",
    "RunLogError",
    "RunLogEvent",
    "RunLogWriter",
    "event_type_counts",
    "events_of_type",
    "load_run_log",
    "missing_event_types",
]
