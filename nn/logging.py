from copy import deepcopy
from typing import Any

from torch import Tensor


class MetricRegistry:
    def __init__(self):
        self._registered_metrics = {}

    def add(self, key: str, value: Any):
        if isinstance(value, Tensor):
            if value.ndim == 0:
                self._registered_metrics[key] = value.detach().item()
            else:
                self._registered_metrics[key] = value.detach().mean().item()
        else:
            self._registered_metrics[key] = value

    def retrieve(self):
        to_return = deepcopy(self._registered_metrics)
        self._registered_metrics = {}
        return to_return
