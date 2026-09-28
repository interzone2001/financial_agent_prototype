import pytest

from tests.ws1_market.helpers import FakeClock


@pytest.fixture
def clock():
    return FakeClock()
