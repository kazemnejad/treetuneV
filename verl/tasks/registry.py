# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

__all__ = ["register", "get_task_cls"]

TASK_REGISTRY = {}


def register(name):
    """Decorator to register a task class with a given name.

    Args:
        name: `(str)`
            The name of the task.
    """

    def decorator(cls):
        if name in TASK_REGISTRY and TASK_REGISTRY[name] != cls:
            raise ValueError(f"Task {name} has already been registered: {TASK_REGISTRY[name]} vs {cls}")
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


def get_task_cls(name):
    """Get the task class with a given name.

    Args:
        name: `(str)`
            The name of the task.

    Returns:
        `(type)`: The task class.
    """
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name}")
    return TASK_REGISTRY[name]
