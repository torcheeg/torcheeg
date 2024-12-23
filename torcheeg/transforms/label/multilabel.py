from typing import Dict, List, Optional, Callable
from ..base_transform import LabelTransform

class Multilabel(LabelTransform):
    r'''
    Apply multiple label transforms independently and combine their results into a list.
    
    .. code-block:: python

        from torcheeg import transforms

        t = transforms.Multilabel([
            transforms.Compose([
                transforms.Select('valence'),
                transforms.Binary(5.0)
            ]),
            transforms.Select('subject_id')
        ])
        t(y={'valence': 4.5, 'arousal': 5.5, 'subject_id': 7})['y']
        >>> [0, 7]

    Args:
        transforms (list): A list of transforms to be applied independently.

    .. automethod:: __call__
    '''
    def __init__(self, transforms: List[LabelTransform], type_fn: Optional[Callable] = list):
        super(Multilabel, self).__init__()
        self.transforms = transforms
        self.type_fn = type_fn

    def __call__(self, *args, y: Dict, **kwargs) -> Dict:
        r'''
        Args:
            y (dict): A dictionary containing the input labels.
            
        Returns:
            dict: A dictionary containing the processed labels as a list.
        '''
        return super().__call__(*args, y=y, **kwargs)

    def apply(self, y: Dict, **kwargs) -> List:
        results = []
        for transform in self.transforms:
            transformed = transform(y=y)
            results.append(transformed['y'])
        return self.type_fn(results)

    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{
            'transforms': self.transforms
        })

if __name__ == '__main__':
    from torcheeg import transforms

    t = Multilabel([
        transforms.Compose([
            transforms.Select('valence'),
            transforms.Binary(5.0)
        ]),
        transforms.Select('subject_id')
    ])
    print(t(y={'valence': 4.5, 'arousal': 5.5, 'subject_id': 7})['y'])
    #>>> [0, 7]

    t = transforms.Compose([
        transforms.Select(['valence', 'arousal']),
        Multilabel([
            transforms.Binary(5.0),
            transforms.Binary(6.0)
        ])
    ])
    print(t(y={'valence': 4.5, 'arousal': 5.5, 'subject_id': 7})['y'])
    #>>> [0, 7]