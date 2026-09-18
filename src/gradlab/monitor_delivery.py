"""Bounded I/O overlap; episode inference and its RNG remain on the calling thread."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock


class MonitoringDelivery:
    """At most two remote operations and three retained jobs per checkpoint worker.

    Dependency callbacks admit commits only after all their chunks are verified.
    They never occupy an I/O thread waiting for another job in the same pool.
    Three queued chunks plus the chunk being recorded fit the four-chunk spool floor.
    """

    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="monitor-r2")
        self.pending = deque()

    def submit(self, operation, *, after=()):
        while self.pending and (len(self.pending) >= 3 or self.pending[0].done()):
            self.pending.popleft().result()
        result = Future()
        self.pending.append(result)

        def relay(done):
            try:
                result.set_result(done.result())
            except BaseException as error:
                result.set_exception(error)

        def schedule():
            try:
                self.pool.submit(operation).add_done_callback(relay)
            except BaseException as error:
                result.set_exception(error)

        if not after:
            schedule()
        else:
            lock = Lock()
            remaining = len(after)
            failure = None

            def dependency(done):
                nonlocal remaining, failure
                with lock:
                    failure = failure or done.exception()
                    remaining -= 1
                    ready = remaining == 0
                if ready:
                    if failure is not None:
                        result.set_exception(failure)
                    else:
                        schedule()

            for dependency_future in after:
                dependency_future.add_done_callback(dependency)
        return result

    def after_pending(self, operation):
        """Publish a local completion only after all preceding remote commits."""
        return self.submit(operation, after=tuple(self.pending))

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        failure = None
        # Join even after a failure: no queued publication may outlive the worker.
        while self.pending:
            try:
                self.pending.popleft().result()
            except BaseException as error:
                failure = failure or error
        self.pool.shutdown(wait=True)
        if failure is not None and kind is None:
            raise failure
