import queue
import threading
import traceback

from joblib import Parallel, delayed, effective_n_jobs

from .base import BaseRunner, _iter_samples
from .multiprocess import _send


def _produce_records(data, records, output, stop):
    try:
        for record in records:
            if stop.is_set():
                return
            for sample in _iter_samples(data, record):
                if not _send(output, ('sample', sample), stop):
                    return
            if not _send(output, ('end', None), stop):
                return
    except BaseException:
        _send(output, ('error', traceback.format_exc()), stop)


class JoblibRunner(BaseRunner):
    """Stream ordered records with Joblib threads and bounded output queues.

    Use MultiprocessRunner for process-isolated CPU execution.
    """

    def __init__(self, num_worker=0, prefetch=2):
        if isinstance(num_worker, int) and num_worker < 0:
            num_worker = effective_n_jobs(num_worker)
        super().__init__(num_worker=num_worker)
        if not isinstance(prefetch, int) or isinstance(prefetch, bool) or prefetch < 1:
            raise ValueError('prefetch must be a positive integer.')
        self.prefetch = prefetch

    def iterate(self, data, records):
        if not self.num_worker or not records:
            for record in records:
                yield from _iter_samples(data, record)
            return
        count = min(self.num_worker, len(records))
        outputs = [queue.Queue(maxsize=self.prefetch) for _ in range(count)]
        stop = threading.Event()
        errors = []

        def execute():
            try:
                Parallel(n_jobs=count, backend='threading')(
                    delayed(_produce_records)(
                        data, records[i::count], outputs[i], stop)
                    for i in range(count))
            except BaseException:
                errors.append(traceback.format_exc())

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        try:
            for index in range(len(records)):
                output = outputs[index % count]
                while True:
                    try:
                        kind, payload = output.get(timeout=0.1)
                    except queue.Empty:
                        if errors or not thread.is_alive():
                            raise RuntimeError(
                                'Joblib workers stopped unexpectedly: ' + ''.join(errors))
                        continue
                    if kind == 'error':
                        raise RuntimeError('Record worker failed:\n' + payload)
                    if kind == 'end':
                        break
                    yield payload
        finally:
            stop.set()
            thread.join()
