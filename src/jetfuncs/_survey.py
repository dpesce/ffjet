"""
Parameter surveys: evaluate a user function on many JetModel configurations in
parallel worker processes.

Each worker builds its own JetModel from a configuration dictionary and calls the
user's function on it, so all per-model setup (stagnation surface, jet-power
normalization, synchrotron tables, jet intervals, precomputed state) happens inside
the worker.  The compiled kernels are already multi-threaded; the process pool is for
the parts that are not (model construction, Python overhead, short kernel calls on
small grids) and for keeping every core busy when individual models are small.
Threads are divided between the workers so that the machine is not oversubscribed.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from ._core import JetModel


def _worker_init(threads_per_worker):
    if threads_per_worker is not None:
        try:
            import numba

            numba.set_num_threads(int(threads_per_worker))
        except ImportError:
            pass


def _run_one(cfg, func, func_kwargs):
    model = JetModel(**cfg)
    return func(model, **func_kwargs)


def survey(
    configs, func, *, n_workers=None, threads_per_worker=None, mp_context="spawn", func_kwargs=None
):
    """
    Evaluate func(model, **func_kwargs) for every configuration in `configs`, in
    parallel worker processes, and return the list of results in the same order.

    configs:            iterable of dicts of JetModel keyword arguments
    func:               callable taking a JetModel (and func_kwargs) and returning a
                        picklable result, e.g. an SED array; it must be importable by
                        the workers, i.e. defined at module level (with the "spawn"
                        start method the calling script must be guarded by
                        `if __name__ == "__main__":`)
    n_workers:          number of worker processes (default: half the cores)
    threads_per_worker: numba threads per worker (default: cores // n_workers, at least 1)
    mp_context:         multiprocessing start method; "spawn" is the safe choice with
                        threaded numba kernels
    """
    configs = list(configs)
    if func_kwargs is None:
        func_kwargs = {}
    ncpu = os.cpu_count() or 1
    if n_workers is None:
        n_workers = max(1, ncpu // 2)
    n_workers = max(1, min(int(n_workers), len(configs))) if configs else 1
    if threads_per_worker is None:
        threads_per_worker = max(1, ncpu // n_workers)
    if n_workers == 1:
        _worker_init(threads_per_worker)
        return [_run_one(cfg, func, func_kwargs) for cfg in configs]
    ctx = multiprocessing.get_context(mp_context)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(threads_per_worker,),
    ) as pool:
        futures = [pool.submit(_run_one, cfg, func, func_kwargs) for cfg in configs]
        return [f.result() for f in futures]
