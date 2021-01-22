import os
import subprocess
from pathlib import Path

this_dir = Path(__file__).absolute().parent
examples_dir = this_dir / ".." / "examples"

EXAMPLES_WITH_KWARGS = {
    "keras_cifar.py": [],
    "keras_mnist_diffpriv.py": [],
    "keras_xray.py": [],
    "pytorch_cifar.py": [],
    "pytorch_mnist_diffpriv.py": [],
    "pytorch_xray.py": [],
    "keras_fraud.py": [str(this_dir / "data" / "fraud")],
    "keras_mnist.py": [],
    "mli_fraud.py": [str(this_dir / "data" / "fraud")],
    "pytorch_covid.py": [],
    "pytorch_mnist.py": [],
    "run_demo.py": []
}

IGNORED = ["run_demo.py", "keras_xray.py", "pytorch_xray.py",
           "pytorch_covid.py", "keras_cifar.py", "pytorch_cifar.py"]


# todo: cifar and mnist downloads need to be cached in github action
def test_examples():
    env = os.environ
    env["MPLBACKEND"] = "agg"
    env["COLEARN_EXAMPLES_TEST"] = "1"
    for script, kwargs in EXAMPLES_WITH_KWARGS.items():
        print(script)
        if script in IGNORED:
            print(f"Ignored {script}")
            continue
        output = subprocess.run(["python", examples_dir / script] + kwargs,
                                capture_output=True,
                                env=env,
                                # timeout=10
                                )
        try:
            output.check_returncode()
        except subprocess.CalledProcessError:
            print("FAIL")
            print(output.stdout,
                  output.stderr)


def test_all_examples_included():
    examples_list = set(x.name for x in examples_dir.glob('*'))
    assert examples_list == set(EXAMPLES_WITH_KWARGS.keys())


if __name__ == "__main__":
    test_all_examples_included()
    test_examples()
