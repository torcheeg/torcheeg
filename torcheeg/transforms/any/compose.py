from typing import Any

from ..base_transform import BaseTransform


class Compose(BaseTransform):
    """Apply callables in order, passing each output to the next callable."""

    def __init__(self, transforms):
        steps = []
        for transform in transforms:
            if not callable(transform):
                raise TypeError('Every transform must be callable.')
            if isinstance(transform, Compose):
                steps.extend(transform.transforms)
            else:
                steps.append(transform)
        self.transforms = tuple(steps)

    def __call__(self, x: Any) -> Any:
        for transform in self.transforms:
            x = transform(x)
        return x

    @property
    def repr_body(self):
        return {'transforms': self.transforms}
