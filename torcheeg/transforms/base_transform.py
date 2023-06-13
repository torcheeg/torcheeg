# https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/transforms_interface.py

from typing import Callable, Dict, Union, List


class BaseTransform:
    def __init__(self):
        self._additional_targets: Dict[str, str] = {}

    def __call__(self, *args, **kwargs) -> Dict[str, any]:
        if args:
            raise KeyError("Please pass data as named parameters.")
        res = {}

        params = self.get_params()

        if self.targets_as_params:
            assert all(key in kwargs
                       for key in self.targets_as_params), "{} requires {}".format(self.__class__.__name__,
                                                                                   self.targets_as_params)
            targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
            params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
            params.update(params_dependent_on_targets)

        for key, arg in kwargs.items():
            if not arg is None:
                target_function = self._get_target_function(key)
                res[key] = target_function(arg, **params)
        return res

    @property
    def targets_as_params(self) -> List[str]:
        return []

    def get_params(self) -> Dict:
        return {}

    def get_params_dependent_on_targets(self, params: Dict[str, any]) -> Dict[str, any]:
        return {}

    def _get_target_function(self, key: str) -> Callable:
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, key)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def add_targets(self, additional_targets: Dict[str, str]):
        self._additional_targets = additional_targets

    @property
    def targets(self) -> Dict[str, Callable]:
        raise NotImplementedError("Method targets is not implemented in class " + self.__class__.__name__)

    def apply(self, *args, **kwargs) -> any:
        raise NotImplementedError("Method apply is not implemented in class " + self.__class__.__name__)

    @property
    def repr_body(self) -> Dict:
        return {}

    def __repr__(self) -> str:
        # init info
        format_string = self.__class__.__name__ + '('
        for i, (k, v) in enumerate(self.repr_body.items()):
            # line end
            if i:
                format_string += ', '
            # str param
            if isinstance(v, str):
                format_string += f"{k}='{v}'"
            else:
                format_string += f"{k}={v}"
        format_string += ')'
        return format_string

class EEGTransform(BaseTransform):
    def __init__(self, apply_to_baseline: bool = False):
        super(EEGTransform, self).__init__()
        self.apply_to_baseline = apply_to_baseline
        if apply_to_baseline:
            self.add_targets({'baseline': 'eeg'})

    @property
    def targets(self):
        return {"eeg": self.apply}

    def apply(self, eeg: any, baseline: Union[any, None] = None, **kwargs) -> any:
        raise NotImplementedError("Method apply is not implemented in class " + self.__class__.__name__)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'apply_to_baseline': self.apply_to_baseline})


class LabelTransform(BaseTransform):
    @property
    def targets(self):
        return {"y": self.apply}

    def apply(self, y: any, **kwargs) -> any:
        raise NotImplementedError("Method apply is not implemented in class " + self.__class__.__name__)