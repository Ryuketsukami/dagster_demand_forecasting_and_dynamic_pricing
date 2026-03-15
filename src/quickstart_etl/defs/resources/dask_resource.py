"""
Dask resource for large-scale parallel backfills.

Usage in an asset:
    def my_asset(context, dask: DaskResource):
        with dask.get_client() as client:
            future = client.submit(some_fn, arg)
            result = future.result()
"""

import contextlib

from dagster import ConfigurableResource
from pydantic import Field


class DaskResource(ConfigurableResource):
    """Provides a dask.distributed Client backed by a LocalCluster.

    Suitable for local backfill runs and development. For production
    scale-out, swap the LocalCluster for a remote scheduler address.

    Must be used as a context manager to ensure the cluster is closed:
        with dask_resource.get_client() as client:
            ...
    """

    n_workers: int = Field(default=4, description="Number of Dask worker processes")
    threads_per_worker: int = Field(default=2, description="Threads per worker")
    memory_limit: str = Field(default="2GiB", description="Memory cap per worker (e.g. '2GiB')")

    @contextlib.contextmanager
    def get_client(self):
        """Yield a dask.distributed.Client; closes cluster and client on exit."""
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
        )
        client = Client(cluster)
        try:
            yield client
        finally:
            client.close()
            cluster.close()


dask_resource = DaskResource()
