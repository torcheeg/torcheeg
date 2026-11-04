import multiprocessing
import pickle
import queue
import traceback

from .base import BaseRunner, _iter_samples


def _send(output, message, stop):
    while not stop.is_set():
        try:
            output.put(message, timeout=0.1)
            return True
        except queue.Full:
            pass
    return False


def _produce_records(data, records, output, stop):
    try:
        for record in records:
            if stop.is_set():
                return
            for sample in _iter_samples(data, record):
                # Serialize before enqueueing so buffer reuse cannot race a feeder.
                if not _send(output, ('sample', pickle.dumps(sample)), stop):
                    return
            if not _send(output, ('end', None), stop):
                return
    except BaseException:
        _send(output, ('error', traceback.format_exc()), stop)


class MultiprocessRunner(BaseRunner):
    """Stream records in source order using bounded per-worker queues."""

    def __init__(self, num_worker=0, multiprocessing_context='spawn', prefetch=2):
        super().__init__(num_worker=num_worker)
        if not isinstance(prefetch, int) or isinstance(prefetch, bool) or prefetch < 1:
            raise ValueError('prefetch must be a positive integer.')
        self.multiprocessing_context = multiprocessing_context
        self.prefetch = prefetch

    def iterate(self, data, records):
        if not self.num_worker:
            for record in records:
                yield from _iter_samples(data, record)
            return
        if not records:
            return
        context = multiprocessing.get_context(self.multiprocessing_context)
        stop = context.Event()
        count = min(self.num_worker, len(records))
        workers = []
        outputs = []
        try:
            for index in range(count):
                output = context.Queue(maxsize=self.prefetch)
                outputs.append(output)
                worker = context.Process(target=_produce_records,
                                         args=(data, records[index::count], output, stop))
                worker.start()
                workers.append(worker)
            for index in range(len(records)):
                output = outputs[index % count]
                worker = workers[index % count]
                while True:
                    try:
                        kind, payload = output.get(timeout=0.1)
                    except queue.Empty:
                        if not worker.is_alive():
                            raise RuntimeError(
                                f'Worker exited without completing record (exitcode={worker.exitcode}).')
                        continue
                    if kind == 'error':
                        raise RuntimeError('Record worker failed:\n' + payload)
                    if kind == 'end':
                        break
                    yield pickle.loads(payload)
        finally:
            stop.set()
            for worker in workers:
                worker.join(timeout=1)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            for output in outputs:
                output.close()
                output.cancel_join_thread()
