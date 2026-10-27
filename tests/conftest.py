"""Release-only availability gate for tests requiring undistributed data."""
import pytest


@pytest.fixture(autouse=True)
def _participant_data_availability(request):
    if request.node.get_closest_marker("requires_participant_data") is None:
        return
    from capclaims.conditional_behavior.source import DEFAULT_SOURCE, POLICY, REGISTRIES
    if not all(path.is_file() for path in (DEFAULT_SOURCE, POLICY, *REGISTRIES.values())):
        pytest.skip("private participant source, policy, and registries are not redistributed")
