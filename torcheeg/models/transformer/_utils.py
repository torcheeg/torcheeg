def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')


def _probability(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ValueError(f'{name} must be in [0, 1).')


