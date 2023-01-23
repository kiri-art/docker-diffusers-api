import pytest
import os
from .lib import startContainer, get_free_port, DOCKER_GW_IP


@pytest.fixture(autouse=True, scope="session")
def my_fixture():
    # setup_stuff
    print("session start")

    # newCache = not os.getenv("DDA_https_proxy")
    newCache = False

    if newCache:
        squid_port = get_free_port()
        http_port = get_free_port()
        container, stop = startContainer(
            "gadicc/squid-ssl-zero",
            ports={3128: squid_port, 3129: http_port},
        )
        os.environ["DDA_http_proxy"] = f"http://{DOCKER_GW_IP}:{squid_port}"
        os.environ["DDA_https_proxy"] = os.environ["DDA_http_proxy"]
        # TODO, code in getDDA to download cert

    yield
    # teardown_stuff
    print("session end")
    if newCache:
        stop()
