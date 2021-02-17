import os
import subprocess
from pathlib import Path
from typing import List, Dict, Sequence

import pytest

REPO_ROOT = Path(__file__).absolute().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

GITHUB_ACTION = bool(os.getenv("GITHUB_ACTION", ""))

if GITHUB_ACTION:
    COLEARN_DATA_DIR = Path("/pvc-data/")
    TFDS_DATA_DIR = str(COLEARN_DATA_DIR / "tensorflow_datasets")
    PYTORCH_DATA_DIR = str(COLEARN_DATA_DIR / "pytorch_datasets")

else:
    COLEARN_DATA_DIR = Path(
        os.getenv("COLEARN_DATA_DIR",
                  os.path.expanduser(os.path.join('~', 'datasets'))))

    TFDS_DATA_DIR = os.getenv("TFDS_DATA_DIR",
                              str(os.path.expanduser(os.path.join('~', "tensorflow_datasets"))))
    PYTORCH_DATA_DIR = os.getenv("PYTORCH_DATA_DIR",
                                 str(os.path.expanduser(os.path.join('~', "pytorch_datasets"))))

FRAUD_DATA_DIR = COLEARN_DATA_DIR / "ieee-fraud-detection"
XRAY_DATA_DIR = COLEARN_DATA_DIR / "chest_xray"
COVID_DATA_DIR = COLEARN_DATA_DIR / "covid"

STANDARD_DEMO_ARGS: List[str] = ["-p", "1", "-n", "3"]

EXAMPLES_WITH_KWARGS = [
    ("keras_cifar.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    ("keras_fraud.py", [FRAUD_DATA_DIR], {}),
    ("keras_mnist.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    ("keras_mnist_diffpriv.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    ("keras_xray.py", [XRAY_DATA_DIR], {}),
    ("mli_fraud.py", [FRAUD_DATA_DIR], {}),
    ("mli_random_forest_iris.py", [], {}),
    ("pytorch_cifar.py", [], {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    ("pytorch_covid.py", [COVID_DATA_DIR], {}),
    ("pytorch_mnist.py", [], {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    ("pytorch_mnist_diffpriv.py", [], {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    ("pytorch_xray.py", [XRAY_DATA_DIR], {}),
    ("run_demo.py", ["-t", "PYTORCH_XRAY", "-d", str(XRAY_DATA_DIR / "train")] + STANDARD_DEMO_ARGS, {}),
    ("run_demo.py", ["-t", "KERAS_MNIST"] + STANDARD_DEMO_ARGS, {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    ("run_demo.py", ["-t", "KERAS_CIFAR10"] + STANDARD_DEMO_ARGS, {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    ("run_demo.py", ["-t", "PYTORCH_COVID_XRAY", "-d", str(COVID_DATA_DIR)] + STANDARD_DEMO_ARGS, {}),
    ("run_demo.py", ["-t", "FRAUD", "-d", str(FRAUD_DATA_DIR)] + STANDARD_DEMO_ARGS, {})
]

IGNORED: List[str] = []


@pytest.mark.parametrize("script,cmd_line,test_env", EXAMPLES_WITH_KWARGS)
@pytest.mark.slow
def test_a_colearn_example(script: str, cmd_line: List[str], test_env: Dict[str, str]):
    env = os.environ
    env["MPLBACKEND"] = "agg"  # disable interacitve plotting
    env["COLEARN_EXAMPLES_TEST"] = "1"  # enables test mode, which sets n_rounds=1
    env.update(test_env)
    print("Additional envvars:", test_env)

    if script in IGNORED:
        pytest.skip(f"Example {script} marked as IGNORED")

    full_cmd: Sequence = ["python", str(EXAMPLES_DIR / script)] + cmd_line
    print("Full command", full_cmd)
    subprocess.run(full_cmd,
                   env=env,
                   timeout=20 * 60,
                   check=True
                   )


def test_all_examples_included():
    examples_list = {x.name for x in EXAMPLES_DIR.glob('*')}
    assert examples_list == {x[0] for x in EXAMPLES_WITH_KWARGS}
