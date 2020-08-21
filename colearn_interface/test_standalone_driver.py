from colearn_interface.standalone_driver import add


def test_sum():
    assert add(1,2) == 5

if __name__ == "__main__":
    # execute only if run as a script
    test_sum()
