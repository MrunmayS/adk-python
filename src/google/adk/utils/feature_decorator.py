# Copyright 2025 Google LLC
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

from __future__ import annotations

import functools
from typing import Callable
from typing import cast
from typing import TypeVar
from typing import Union
import warnings

T = TypeVar("T", bound=Union[Callable, type])


def _make_feature_decorator(
    *, label: str, default_message: str
) -> Callable[[str], Callable[[T], T]]:
  def decorator_factory(message: str = default_message) -> Callable[[T], T]:
    def decorator(obj: T) -> T:
      obj_name = getattr(obj, "__name__", type(obj).__name__)
      warn_msg = f"[{label.upper()}] {obj_name}: {message}"

      if isinstance(obj, type):  # decorating a class
        orig_init = obj.__init__

        @functools.wraps(orig_init)
        def new_init(self, *args, **kwargs):
          warnings.warn(warn_msg, category=UserWarning, stacklevel=2)
          return orig_init(self, *args, **kwargs)

        obj.__init__ = new_init  # type: ignore[attr-defined]
        return cast(T, obj)

      elif callable(obj):  # decorating a function or method

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
          warnings.warn(warn_msg, category=UserWarning, stacklevel=2)
          return obj(*args, **kwargs)

        return cast(T, wrapper)

      else:
        raise TypeError(
            f"@{label} can only be applied to classes or callable objects"
        )

    return decorator

  return decorator_factory


working_in_progress = _make_feature_decorator(
    label="WIP",
    default_message=(
        "This feature is a work in progress and may be incomplete or unstable."
    ),
)
"""Mark a class or function as a work in progress.

Sample usage:

```
@working_in_progress("This feature is not ready for production use.")
def my_wip_function():
  pass
```
"""

experimental = _make_feature_decorator(
    label="EXPERIMENTAL",
    default_message=(
        "This feature is experimental and may change or be removed in future"
        " versions without notice. It may introduce breaking changes at any"
        " time."
    ),
)
"""Mark a class or a function as an experimental feature.

Sample usage:

```
@experimental("This API may have breaking change in the future.")
class ExperimentalClass:
  pass
```
"""
