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

__all__ = ["register", "get_trainer_cls"]

TRAINER_REGISTRY = {}


def register(name):
    """Decorator to register a trainer class with a given name.

    Args:
        name: `(str)`
            The name of the trainer.
    """

    def decorator(cls):
        if name in TRAINER_REGISTRY and TRAINER_REGISTRY[name] != cls:
            raise ValueError(f"Trainer {name} has already been registered: {TRAINER_REGISTRY[name]} vs {cls}")
        TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


def get_trainer_cls(name):
    """Get the trainer class with a given name.

    Args:
        name: `(str)`
            The name of the trainer.

    Returns:
        `(type)`: The trainer class.
    """
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown trainer: {name}")
    return TRAINER_REGISTRY[name]
