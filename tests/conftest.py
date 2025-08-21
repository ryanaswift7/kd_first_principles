# Add this at the top of your test file or in conftest.py
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: integration tests that require external resources"
    )
    config.addinivalue_line(
        "markers", "slow: tests that are slow to run"
    )