from ..base_transform import BaseTransform


class Lambda(BaseTransform):
    """Apply a user-provided callable to one value."""

    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, x):
        return self.lambd(x)

    @property
    def repr_body(self):
        return {'lambd': self.lambd}
