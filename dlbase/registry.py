"""Central registries for dataset / model / task plugins (see ``dlbase/plugin_bootstrap``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Type

from omegaconf import DictConfig


class TaskModule(ABC):
    """Abstract task contract: Hydra config in, training side effects out (no d/m/t implementations)."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, cfg: DictConfig) -> None:
        ...



@dataclass(frozen=True)
class DatasetPlugin:
    """``dataset.name`` → setup + checks + summary note."""

    setup: Callable[[Any, DictConfig, Any], None]
    assert_task_supported: Callable[[DictConfig], None]
    eval_split_note: Callable[[Any], str]


@dataclass(frozen=True)
class ModelPlugin:
    """``model.name`` → compatibility gate + weight construction."""

    assert_compatible: Callable[[DictConfig], None]
    construct: Callable[[DictConfig], Any]


@dataclass(frozen=True)
class TaskPlugin:
    """``task.name`` → :class:`TaskModule` implementation."""

    task_class: Type[TaskModule]


_DATASET_REGISTRY: dict[str, DatasetPlugin] = {}
_DATASET_ALIASES: dict[str, str] = {}
_MODEL_REGISTRY: dict[str, ModelPlugin] = {}
_TASK_REGISTRY: dict[str, TaskPlugin] = {}
_PLUGINS_BOOTSTRAPPED = False


def _ensure_plugins() -> None:
    global _PLUGINS_BOOTSTRAPPED
    if _PLUGINS_BOOTSTRAPPED:
        return
    import dlbase.plugin_bootstrap as plugin_bootstrap  # noqa: PLC0415 — avoid import cycle at module load

    plugin_bootstrap.ensure_plugins_loaded()
    _PLUGINS_BOOTSTRAPPED = True


def register_dataset(
    name: str,
    setup: Callable[[Any, DictConfig, Any], None],
    assert_task_supported: Callable[[DictConfig], None],
    eval_split_note: Callable[[Any], str],
) -> None:
    if name in _DATASET_REGISTRY:
        raise ValueError(f"register_dataset: {name!r} is already registered")
    _DATASET_REGISTRY[name] = DatasetPlugin(
        setup=setup,
        assert_task_supported=assert_task_supported,
        eval_split_note=eval_split_note,
    )


def register_dataset_alias(alias: str, canonical: str) -> None:
    if alias in _DATASET_ALIASES:
        raise ValueError(f"register_dataset_alias: {alias!r} already mapped to {_DATASET_ALIASES[alias]!r}")
    _DATASET_ALIASES[alias] = canonical


def resolve_dataset_name(name: str) -> str:
    return _DATASET_ALIASES.get(name, name)


def register_model(
    name: str,
    assert_compatible: Callable[[DictConfig], None],
    construct: Callable[[DictConfig], Any],
) -> None:
    if name in _MODEL_REGISTRY:
        raise ValueError(f"register_model: {name!r} is already registered")
    _MODEL_REGISTRY[name] = ModelPlugin(assert_compatible=assert_compatible, construct=construct)


def register_task(name: str, task_class: Type[TaskModule]) -> None:
    if name in _TASK_REGISTRY:
        raise ValueError(f"register_task: {name!r} is already registered")
    _TASK_REGISTRY[name] = TaskPlugin(task_class=task_class)


def get_dataset_plugin(name: str) -> DatasetPlugin:
    _ensure_plugins()
    key = resolve_dataset_name(str(name))
    if key not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset.name: {name!r}")
    return _DATASET_REGISTRY[key]


def get_model_plugin(name: str) -> ModelPlugin:
    _ensure_plugins()
    key = str(name)
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model.name: {key!r}")
    return _MODEL_REGISTRY[key]


def get_task_plugin(name: str) -> TaskPlugin:
    _ensure_plugins()
    key = str(name)
    if key not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task.name={key!r}. Add tasks/{key}/ and register in tasks.plugins."
        )
    return _TASK_REGISTRY[key]


def list_registered_dataset_keys() -> list[str]:
    """Canonical dataset keys (aliases map to these; alias keys are not listed)."""
    _ensure_plugins()
    return sorted(_DATASET_REGISTRY.keys())


def list_registered_model_keys() -> list[str]:
    _ensure_plugins()
    return sorted(_MODEL_REGISTRY.keys())


def list_registered_task_keys() -> list[str]:
    _ensure_plugins()
    return sorted(_TASK_REGISTRY.keys())
