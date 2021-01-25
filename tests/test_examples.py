import logging
import os
import subprocess
from pathlib import Path

import pytest

this_dir = Path(__file__).absolute().parent
examples_dir = this_dir / ".." / "examples"

MY_DATASETS = this_dir / ".." / ".." / "datasets"
# MY_DATASETS = Path("/pvc-data/")

TFDS_DATA_DIR = os.getenv("TFDS_DATA_DIR", str(MY_DATASETS / "tensorflow_datasets"))
PYTORCH_DATA_DIR = os.getenv("PYTORCH_DATA_DIR", str(MY_DATASETS / "pytorch_datasets"))

logger = logging.getLogger()
logger.info(f"MY_DATASETS {MY_DATASETS}")
logger.info(f"TFDS_DATA_DIR {TFDS_DATA_DIR}")
logger.info(f"PYTORCH_DATA_DIR {PYTORCH_DATA_DIR}")


EXAMPLES_WITH_KWARGS = [
    ("keras_cifar.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    ("keras_fraud.py", [str(MY_DATASETS / "ieee-fraud-detection")], {}),
    ("keras_mnist.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    ("keras_mnist_diffpriv.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    ("keras_xray.py", [str(MY_DATASETS / "chest_xray")], {}),
    ("mli_fraud.py", [str(MY_DATASETS / "ieee-fraud-detection")], {}),
    ("pytorch_cifar.py", [], {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    ("pytorch_covid.py", [str(MY_DATASETS / "covid")], {}),
    ("pytorch_mnist.py", [],  {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    ("pytorch_mnist_diffpriv.py", [], {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    ("pytorch_xray.py", [str(MY_DATASETS / "chest_xray")], {}),
    ("run_demo.py", [], {})
]

IGNORED = ["run_demo.py", ]


@pytest.mark.parametrize("script,cmd_line,test_env", EXAMPLES_WITH_KWARGS)
def test_a_colearn_example(script, cmd_line, test_env):
    env = os.environ
    env["MPLBACKEND"] = "agg"
    env["COLEARN_EXAMPLES_TEST"] = "1"
    env.update(test_env)

    print(script)
    if script in IGNORED:
        print(f"Ignored {script}")
        pytest.skip(f"Example {script} marked as IGNORED")

    output = subprocess.run(["python", examples_dir / script] + cmd_line,
                            capture_output=True,
                            env=env,
                            timeout=20*60
                            )
    try:
        output.check_returncode()
    except subprocess.CalledProcessError:
        print("FAIL")
        print(output.stdout,
              output.stderr)
        raise


def test_all_examples_included():
    examples_list = set(x.name for x in examples_dir.glob('*'))
    assert examples_list == set([ x[0] for x in EXAMPLES_WITH_KWARGS])
