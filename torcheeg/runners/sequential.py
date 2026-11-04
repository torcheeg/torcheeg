from .base import BaseRunner, _iter_samples


class SequentialRunner(BaseRunner):
    """Process records and yield each sample in the calling process."""

    def __init__(self):
        super().__init__(num_worker=0)

    def iterate(self, data, records):
        for record in records:
            yield from _iter_samples(data, record)
