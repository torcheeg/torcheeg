class BaseTransform:
    """Transform one value into another value."""

    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        raise NotImplementedError(
            "Method apply is not implemented in class " + self.__class__.__name__)

    @property
    def repr_body(self):
        return {}

    def __repr__(self):
        values = ', '.join(f'{name}={value!r}' for name,
                           value in self.repr_body.items())
        return f'{type(self).__name__}({values})'


class EEGTransform(BaseTransform):
    """Transform an EEG array or tensor."""


class LabelTransform(BaseTransform):
    """Transform a metadata value or label."""
