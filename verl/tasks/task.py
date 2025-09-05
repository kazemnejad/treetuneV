import hashlib
import inspect
import logging
import marshal
import os
import pickle
import tempfile
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from verl.base_config import BaseConfig
from verl.tasks.registry import get_task_cls
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fs import is_non_local
from verl.utils.hdfs_io import copy, makedirs

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _fingerprint_class_object(cls: type) -> str:
    """MD5 fingerprint of a class's implementation and simple attributes.

    - Stable across runs on the same Python version.
    - Includes module + qualname.
    - Includes code objects of functions, staticmethods, classmethods, and properties (via marshal).
    - Includes function defaults, kwdefaults, and annotations.
    - Includes simple data attributes (str/int/float/bool/None/bytes and simple tuples of those).
    """
    h = hashlib.md5()
    h.update(f"{cls.__module__}.{cls.__qualname__}".encode("utf-8"))

    def _stable_update_value(v):
        # Deterministic, type-tagged updates for simple data.
        if v is None:
            h.update(b"N")
        elif isinstance(v, bool):
            h.update(b"b1" if v else b"b0")
        elif isinstance(v, int):
            h.update(b"I")
            h.update(str(v).encode("utf-8"))
        elif isinstance(v, float):
            h.update(b"F")
            h.update(repr(v).encode("utf-8"))  # repr is stable for floats
        elif isinstance(v, str):
            h.update(b"S")
            h.update(v.encode("utf-8"))
        elif isinstance(v, bytes):
            h.update(b"B")
            h.update(v)
        elif isinstance(v, tuple):
            h.update(b"T")
            for e in v:
                _stable_update_value(e)
        else:
            # Fallback: avoid nondeterministic repr; use type name only.
            h.update(b"U")
            h.update(type(v).__name__.encode("utf-8"))

    def _update_function(fn):
        co = getattr(fn, "__code__", None)
        if isinstance(co, types.CodeType):
            # Marshal serialization is what .pyc uses; stable across runs of same Python version.
            h.update(b"C")
            h.update(marshal.dumps(co))

        # Include call signature-affecting metadata deterministically.
        defaults = getattr(fn, "__defaults__", None)
        if defaults is not None:
            h.update(b"D")
            _stable_update_value(defaults)

        kwdefaults = getattr(fn, "__kwdefaults__", None)
        if kwdefaults:
            h.update(b"K")
            for k in sorted(kwdefaults):
                h.update(b"k")
                h.update(str(k).encode("utf-8"))
                _stable_update_value(kwdefaults[k])

        ann = getattr(fn, "__annotations__", None)
        if ann:
            h.update(b"A")
            for k in sorted(ann):
                h.update(b"a")
                h.update(str(k).encode("utf-8"))
                _stable_update_value(ann[k])

    for name, attr in sorted(cls.__dict__.items()):
        if inspect.isfunction(attr):
            _update_function(attr)
            continue
        if isinstance(attr, (staticmethod, classmethod)):
            _update_function(attr.__func__)
            continue
        if isinstance(attr, property):
            for f in (attr.fget, attr.fset, attr.fdel):
                if f is not None:
                    _update_function(f)
            continue

        # Simple data attributes: value or tuple of simple values
        if isinstance(attr, (str, int, float, bool, type(None), bytes)):
            h.update(b"X")
            h.update(name.encode("utf-8"))
            h.update(b"=")
            _stable_update_value(attr)
        elif isinstance(attr, tuple) and all(isinstance(e, (str, int, float, bool, type(None), bytes)) for e in attr):
            h.update(b"Y")
            h.update(name.encode("utf-8"))
            h.update(b"=")
            _stable_update_value(attr)

    return h.hexdigest()


class Split(str, Enum):
    """Standard dataset splits used across tasks."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @classmethod
    def from_str(cls, split_str: str) -> "Split":
        if split_str == "train":
            return cls.TRAIN
        elif split_str == "validation":
            return cls.VALIDATION
        elif split_str == "test":
            return cls.TEST
        else:
            raise ValueError(f"Invalid split: {split_str}")


@dataclass
class LoadingParams:
    args: list[Any]
    kwargs: dict[str, Any]


@dataclass
class TaskConfig(BaseConfig):
    _mutable_fields = {"extra", "loading_params"}

    name: str = MISSING
    loading_params: LoadingParams = LoadingParams(args=[], kwargs={})
    num_dataset_workers: int = 4
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    data_source: Optional[str] = None
    max_samples: Optional[int] = None
    data_proportion: Optional[float] = None
    shuffle_before_sampling: bool = True
    remove_useless_columns: bool = True
    seed: int = 42
    val_sampling_params: Optional[dict[str, float | int]] = None

    def __post_init__(self):
        assert self.name is not None, "Task name is required"

        if self.max_samples is not None and self.data_proportion is not None:
            raise ValueError("Only one of max_samples or data_proportion can be set")

        if not isinstance(self.loading_params, LoadingParams):
            self.loading_params = LoadingParams(**self.loading_params)


class Task:
    def __init__(self, config: DictConfig, cache_dir: Optional[str] = None):
        self._frozen_config = OmegaConf.to_container(config, resolve=True)
        self.config = omega_conf_to_dataclass(config)
        self.cache_dir = os.path.expanduser(cache_dir or "~/.cache/verl/tasks")

    def get_name(self) -> str:
        return self.__class__.__name__

    def _load_dataset_from_config(self, split: Split):
        """Load dataset from config data paths for the given split.

        Args:
            split: The dataset split to load

        Returns:
            The loaded dataset
        """
        loading_params = self.config.loading_params
        dataset = load_dataset(*loading_params.args, **loading_params.kwargs, num_proc=self.config.num_dataset_workers)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split.value]
        return dataset

    def _sample_dataset(self, dataset: Dataset) -> Dataset:
        if self.config.shuffle_before_sampling:
            dataset = dataset.shuffle(seed=self.config.seed)

        if self.config.max_samples is not None:
            dataset = dataset.take(self.config.max_samples)
        elif self.config.data_proportion is not None:
            dataset = dataset.take(int(len(dataset) * self.config.data_proportion))
        return dataset

    def _remove_useless_columns(self, dataset: Dataset) -> Dataset:
        if self.config.remove_useless_columns:
            needed_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
            needed_columns += [f"val_sampling_params.{k}" for k in self.config.val_sampling_params or {}]
            dataset = dataset.remove_columns(set(dataset.column_names) - set(needed_columns))
        return dataset

    def build_dataset(self, split: Split) -> Dataset:
        """Build and return the dataset for the given split.

        Subclasses should implement this method.
        """
        raise NotImplementedError

    def _get_cache_key(self, split: Split) -> str:
        impl_hash = _fingerprint_class_object(self.__class__)

        payload = pickle.dumps(self._frozen_config, protocol=pickle.HIGHEST_PROTOCOL)
        config_hash = hashlib.sha256(payload).hexdigest()[:12]

        return f"{impl_hash}:{config_hash}:{split.value}"

    def get_dataset_parquet_path(self, split: Split) -> str:
        cache_key = self._get_cache_key(split)

        dataset_file = f"{cache_key}.parquet"
        cached_path = os.path.join(self.cache_dir, dataset_file)

        if os.path.exists(cached_path):
            logger.info(f"Found cached dataset for {self}, split={split}")
            return cached_path

        logger.info(f"No cached dataset found for {self}, split={split}, rebuilding...")

        dataset = self.build_dataset(split)

        # Make sure the dataset has the required columns
        datum = dataset[0]
        required_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
        for column in required_columns:
            assert column in datum, f"Dataset for task={self.__class__.__name__} is missing required column: {column}"

        # reward checks
        assert "ground_truth" in datum["reward_model"], (
            f"Dataset for task={self.__class__.__name__} is missing required column: ground_truth"
        )
        assert "style" in datum["reward_model"], (
            f"Dataset for task={self.__class__.__name__} is missing required column: style"
        )

        # extra_info checks
        assert "split" in datum["extra_info"], (
            f"Dataset for task={self.__class__.__name__} is missing required column: split"
        )
        assert "index" in datum["extra_info"], (
            f"Dataset for task={self.__class__.__name__} is missing required column: index"
        )

        if is_non_local(self.cache_dir):
            logger.info(f"Dataset for task={self.__class__.__name__} is non-local, skipping caching")
            cache_dir = tempfile.gettempdir()
            cached_path = os.path.join(cache_dir, dataset_file)
            dataset.to_parquet(cached_path)

            # copy to hdfs
            makedirs(self.cache_dir)
            copy(src=cached_path, dst=self.cache_dir)
        else:
            dataset.to_parquet(cached_path)

        return cached_path

    def __repr__(self) -> str:
        if self.config.data_source is not None:
            return f"{self.__class__.__name__}({self.config.data_source})"
        else:
            return f"{self.__class__.__name__}(name={self.config.name})"


def get_dataset_paths(task_configs: ListConfig, split: Split, cache_dir: Optional[str] = None) -> list[str]:
    """Get the dataset paths for the given task configs and split.

    Args:
        task_configs: The task configs
        split: The dataset split

    Returns:
        The dataset paths
    """
    dataset_paths = []
    for task_config in tqdm(task_configs, desc="Getting dataset paths"):
        task_class = get_task_cls(task_config.name)
        task = task_class(task_config, cache_dir=cache_dir)
        dataset_paths.append(task.get_dataset_parquet_path(split))
    return dataset_paths
