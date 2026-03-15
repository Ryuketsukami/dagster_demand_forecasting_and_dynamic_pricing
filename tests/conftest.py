"""
Pytest configuration for the quickstart_etl test suite.

Sets DAGSTER_HOME to a temporary directory for the duration of the test
session so that Dagster can initialise its local instance storage without
requiring a pre-existing directory on the developer's machine or CI runner.
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def dagster_home(tmp_path_factory):
    """Create a temporary DAGSTER_HOME for the whole test session."""
    home = tmp_path_factory.mktemp("dagster_home")
    os.environ["DAGSTER_HOME"] = str(home)
    yield home
    # tmp_path_factory cleans up automatically after the session
