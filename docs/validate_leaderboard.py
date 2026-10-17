"""Validate leaderboard protocols, metrics and reproducible example references."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlparse
from urllib.request import urlopen


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def text(value):
    return isinstance(value, str) and bool(value.strip())


def positive_integer(value):
    return type(value) is int and value > 0


def hex_digest(value, size):
    return text(value) and re.fullmatch(r'[0-9a-f]{' + str(size) + '}', value) is not None


def https_url(value):
    return text(value) and urlparse(value).scheme == 'https' and bool(urlparse(value).netloc)


def validate(data):
    require(isinstance(data, dict) and data.get('schema_version') == 1, 'Unsupported schema version.')
    protocols = data.get('protocols')
    require(isinstance(protocols, dict) and bool(protocols), 'Protocols must be a nonempty object.')
    for key, protocol in protocols.items():
        require(isinstance(protocol, dict), f'Invalid protocol: {key}')
        for field in ('name', 'dataset', 'task', 'scope', 'preprocessing'):
            require(text(protocol.get(field)), f'{key}: missing {field}.')
        require(https_url(protocol.get('source_url')), f'{key}: invalid data source URL.')
        for field in ('train_windows', 'test_windows', 'epochs', 'seed'):
            value = protocol.get(field)
            require(value is None or (type(value) is int and value >= 0), f'{key}: invalid {field}.')
        for field in ('channels', 'window_samples'):
            value = protocol.get(field)
            require(value is None or positive_integer(value), f'{key}: invalid {field}.')
        rate = protocol.get('sampling_rate')
        require(rate is None or (type(rate) in (int, float) and math.isfinite(rate) and rate > 0), f'{key}: invalid sampling rate.')
        split = protocol.get('split', {})
        require(isinstance(split, dict), f'{key}: invalid split.')
        require(text(split.get('type')), f'{key}: missing split type.')
        if split['type'] == 'fixed_subject_holdout' and (split.get('train_subjects') is not None or split.get('test_subjects') is not None):
            train, test = split.get('train_subjects'), split.get('test_subjects')
            require(isinstance(train, list) and bool(train) and all(text(x) for x in train),
                    f'{key}: invalid training subjects.')
            require(isinstance(test, list) and bool(test) and all(text(x) for x in test),
                    f'{key}: invalid test subjects.')
            require(not (set(train) & set(test)), f'{key}: training and test subjects overlap.')
        for field, total in (('train_class_counts', 'train_windows'), ('test_class_counts', 'test_windows')):
            if protocol.get(field) is not None:
                counts = protocol[field]
                require(isinstance(counts, dict) and bool(counts)
                        and all(positive_integer(x) for x in counts.values()), f'{key}: invalid class counts.')
                require(sum(counts.values()) == protocol[total], f'{key}: class counts do not match windows.')
        for item in protocol.get('data_files') or []:
            require(isinstance(item, dict) and text(item.get('name')) and hex_digest(item.get('sha256'), 64),
                    f'{key}: invalid source-file manifest.')

    results = data.get('results')
    require(isinstance(results, list) and bool(results), 'Results must be a nonempty list.')
    ids = set()
    for result in results:
        require(isinstance(result, dict) and text(result.get('id')), 'A result needs an ID.')
        key = result['id']
        require(key not in ids, f'Duplicate result ID: {key}')
        ids.add(key)
        require(text(result.get('model')), f'{key}: missing model.')
        require(result.get('protocol') in protocols, f'{key}: unknown protocol.')
        metric = result.get('metric', {})
        require(isinstance(metric, dict), f'{key}: invalid metric.')
        require(text(metric.get('name')) and text(metric.get('unit')), f'{key}: missing metric fields.')
        value = metric.get('value')
        require(type(value) in (int, float) and math.isfinite(value), f'{key}: non-finite metric.')
        require(metric.get('direction') in ('higher', 'lower'), f'{key}: invalid metric direction.')
        if metric.get('std') is not None:
            require(type(metric['std']) in (int, float) and math.isfinite(metric['std']) and metric['std'] >= 0, f'{key}: invalid metric deviation.')
        if metric['unit'] == 'fraction':
            require(0 <= value <= 1, f'{key}: fraction metric must be between zero and one.')
        if metric['name'] in ('mse', 'rmse', 'reconstruction_mse'):
            require(value >= 0, f'{key}: squared error must be nonnegative.')
        implementation = result.get('implementation', {})
        require(isinstance(implementation, dict), f'{key}: invalid example reference.')
        require(implementation.get('repository') == 'torcheeg/torcheeg', f'{key}: unexpected example repository.')
        require(hex_digest(implementation.get('commit'), 40), f'{key}: invalid source commit.')
        require(text(implementation.get('example')), f'{key}: missing example path.')
        path = PurePosixPath(implementation['example'])
        require(not path.is_absolute() and '..' not in path.parts
                and path.parts[:1] == ('examples',) and path.suffix == '.py', f'{key}: invalid example path.')
        require(hex_digest(implementation.get('sha256'), 64), f'{key}: missing example checksum.')
        evidence = result.get('evidence', {})
        require(isinstance(evidence, dict), f'{key}: invalid evidence.')
        kind = evidence.get('kind')
        require(kind in ('published', 'measured', 'validation'), f'{key}: invalid evidence kind.')
        if kind == 'published':
            require(https_url(evidence.get('citation_url')), f'{key}: published results need a citation URL.')
        else:
            require(text(evidence.get('command')), f'{key}: missing run command.')
            require(text(evidence.get('recorded_at')), f'{key}: missing run date.')
            datetime.fromisoformat(evidence['recorded_at'].replace('Z', '+00:00'))
            protocol = protocols[result['protocol']]
            for field in ('test_windows', 'channels', 'window_samples'):
                require(positive_integer(protocol.get(field)), f'{key}: measured runs need {field}.')
            require(protocol.get('sampling_rate') is not None, f'{key}: measured runs need sampling rate.')
            for field in ('train_windows', 'epochs', 'seed'):
                require(type(protocol.get(field)) is int and protocol[field] >= 0, f'{key}: measured runs need {field}.')
            if protocol['split']['type'] == 'fixed_subject_holdout':
                require(bool(protocol['split'].get('train_subjects')) and bool(protocol['split'].get('test_subjects')), f'{key}: measured holdouts need subject IDs.')
            environment = result.get('environment', {})
            require(isinstance(environment, dict), f'{key}: invalid environment.')
            require(all(text(environment.get(field)) for field in ('python', 'torch', 'execution_device')),
                    f'{key}: missing run environment.')
            require(bool(protocols[result['protocol']].get('data_files')), f'{key}: missing source-file manifest.')
    return data


def load(path):
    with Path(path).open(encoding='utf-8') as file:
        return validate(json.load(file, object_pairs_hook=unique_object))


def verify_sources(data):
    checked = set()
    for result in data['results']:
        source = result['implementation']
        url = f"https://raw.githubusercontent.com/{source['repository']}/{source['commit']}/{source['example']}"
        if url in checked:
            continue
        with urlopen(url, timeout=30) as response:
            digest = hashlib.sha256(response.read()).hexdigest()
        require(digest == source['sha256'], f'Example checksum mismatch: {url}')
        checked.add(url)
    return len(checked)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path,
                        default=Path(__file__).parent / 'source/_static/leaderboard/results.json')
    parser.add_argument('--verify-sources', action='store_true')
    args = parser.parse_args()
    data = load(args.path)
    print(f"Validated {len(data['results'])} results in {len(data['protocols'])} protocols.")
    if args.verify_sources:
        print(f'Verified {verify_sources(data)} immutable example checksums.')
