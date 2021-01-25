import os
import subprocess
from pathlib import Path
from typing import List, Dict

import pytest

REPO_ROOT = Path(__file__).absolute().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

COLEARN_DATA_DIR = Path(os.getenv("COLEARN_DATA_DIR",
                                  REPO_ROOT / ".." / "datasets"))

TFDS_DATA_DIR = os.getenv("TFDS_DATA_DIR",
                          str(COLEARN_DATA_DIR / "tensorflow_datasets"))
PYTORCH_DATA_DIR = os.getenv("PYTORCH_DATA_DIR",
                             str(COLEARN_DATA_DIR / "pytorch_datasets"))

print("env stuff", COLEARN_DATA_DIR, TFDS_DATA_DIR, PYTORCH_DATA_DIR)

FRAUD_DATA_DIR = COLEARN_DATA_DIR / "ieee-fraud-detection"
XRAY_DATA_DIR = COLEARN_DATA_DIR / "chest_xray"

EXAMPLES_WITH_KWARGS = [
    ("keras_cifar.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    ("keras_fraud.py", [FRAUD_DATA_DIR], {}),
    ("keras_mnist.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    ("keras_mnist_diffpriv.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    ("keras_xray.py", [XRAY_DATA_DIR], {}),
    ("mli_fraud.py", [FRAUD_DATA_DIR], {}),
    ("pytorch_cifar.py", [], {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    ("pytorch_covid.py", [str(COLEARN_DATA_DIR / "covid")], {}),
    ("pytorch_mnist.py", [], {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    ("pytorch_mnist_diffpriv.py", [], {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    ("pytorch_xray.py", [XRAY_DATA_DIR], {}),
    ("run_demo.py", [], {})
]

IGNORED = ["run_demo.py", ]


@pytest.mark.parametrize("script,cmd_line,test_env", EXAMPLES_WITH_KWARGS)
def test_a_colearn_example(script: str, cmd_line: List[str], test_env: Dict[str, str]):
    env = os.environ
    env["MPLBACKEND"] = "agg"  # disable interacitve plotting
    env["COLEARN_EXAMPLES_TEST"] = "1"  # enables test mode, which sets n_rounds=1
    env.update(test_env)

    if script in IGNORED:
        pytest.skip(f"Example {script} marked as IGNORED")

    output = subprocess.run(["python", EXAMPLES_DIR / script] + cmd_line,
                            env=env,
                            timeout=20 * 60
                            )

    output.check_returncode()


def test_all_examples_included():
    examples_list = set(x.name for x in EXAMPLES_DIR.glob('*'))
    assert examples_list == set([x[0] for x in EXAMPLES_WITH_KWARGS])
