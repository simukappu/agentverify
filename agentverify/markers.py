"""Marker definitions for the agentverify pytest plugin.

The ``agentverify`` marker is registered automatically via
:func:`agentverify.plugin.pytest_configure` when the plugin is loaded.

Usage::

    @pytest.mark.agentverify
    def test_my_agent():
        ...

You can selectively run marked tests with::

    pytest -m agentverify
"""

MARKER_NAME = "agentverify"
MARKER_DESCRIPTION = "mark test as an agentverify agent test"
