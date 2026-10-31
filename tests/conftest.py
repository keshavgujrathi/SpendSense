import warnings

import pytest

from spendsense.pipeline import prepare_sample

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def prepared():
    return prepare_sample()
