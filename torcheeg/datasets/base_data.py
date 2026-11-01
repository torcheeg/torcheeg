class BaseData:
    """Define record discovery, reading and sample preprocessing."""

    def __init__(self, transform=None, **configuration):
        removed = {'before_trial', 'after_trial',
                   'after_session', 'after_subject', 'offline_transform',
                   'online_transform', 'label_transform', 'lazy_threshold'}
        invalid = removed.intersection(configuration)
        if invalid:
            raise TypeError('Removed Data arguments: ' +
                            ', '.join(sorted(invalid)))
        self.transform = transform
        self.configuration = dict(configuration)
        self.__dict__.update(configuration)

    def get_records(self):
        return list(self.set_records(**self.configuration))

    def iter_record(self, record):
        parameters = dict(self.configuration, record=record)
        parameters.update(self.read_record(**parameters))
        for sample in self.process_record(**parameters):
            if sample is None:
                continue
            if self.transform is not None:
                sample = self.transform(sample)
            yield sample

    def set_records(self, **kwargs):
        raise NotImplementedError(
            "Method set_records is not implemented in class " + self.__class__.__name__)

    def read_record(self, record, **kwargs):
        raise NotImplementedError(
            "Method read_record is not implemented in class " + self.__class__.__name__)

    @classmethod
    def fake_record(cls, **kwargs):
        raise NotImplementedError(
            "Method fake_record is not implemented in class " + cls.__name__)

    def process_record(self, record, **kwargs):
        raise NotImplementedError(
            "Method process_record is not implemented in class " + self.__class__.__name__)
