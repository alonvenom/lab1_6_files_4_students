import pytest
from main import hello_world
@pytest.mark.order(1)

def test_f(capfd):
    hello_world()
    out, err = capfd.readouterr()
    assert out == "Hello World!\n"