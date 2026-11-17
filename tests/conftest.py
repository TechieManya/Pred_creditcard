import pytest

# Fix for anyio + Windows event loop deadlock with FastAPI TestClient
@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
