import os
import pickle

import numpy as np

_DATA_PATHS = {
    "mnist": "tests/data/mnist/",
    "cifar10": "tests/data/cifar10/"
}


def get_pickle(fname):
    with open(fname, "rb") as f:
        assert f is not None
        data = pickle.load(f)
        assert data is not None
        return data


class FileTester:
    def __init__(self, tolerance=1e-2):
        self._tolerance = tolerance

    def test_object_match(self, s, t):
        if type(s) is not type(t):
            return False, "Type mismatch"
        if isinstance(s, dict):
            if len(s) != len(t):
                return False, "dictionary with different sizes, len(s)={}, len(t)={}".format(len(s), len(t))
            for key in s:
                if key not in t:
                    return False, "dictionary key {} not found in target".format(key)
                status, msg = self.test_object_match(s[key], t[key])
                if not status:
                    return False, "dict test key {} failed: {}".format(key, msg)
            return True, ""
        elif isinstance(s, list):
            if len(s) != len(t):
                return False, f"lists with different sizes, len(s)={len(s)}, len(t)={len(t)}"
            for i, si in enumerate(s):
                status, msg = self.test_object_match(si, t[i])
                if not status:
                    return False, "list test idx {} failed: {}".format(i, msg)
            return True, ""
        elif type(s).__module__ == np.__name__:
            ns = np.linalg.norm(s)
            nt = np.linalg.norm(t)
            if ns == 0 or nt == 0:
                if ns != nt:
                    return False, "zero norm mismatch {}!={}".format(ns, nt)
                return s == t, "zero norm, matrix mismatch"
            dist = np.linalg.norm(s - t) / ns / nt
            return dist < self._tolerance, "matrix mismatch given {} tolerance, (actual distance {})".format(
                self._tolerance, dist)
        else:
            return False, "Not known object type: {}".format(type(s))

    def test_pickle(self, source, target):
        s = get_pickle(source)
        t = get_pickle(target)
        return self.test_object_match(s, t)

    def test(self, test, subpath, result_dir):
        assert test in _DATA_PATHS, "Trying to use a test, which doesn't have data dir"
        root_dir = os.path.join(_DATA_PATHS[test], subpath)
        assert os.path.exists(root_dir), "Dir not found: {}".format(root_dir)
        test_dir = os.path.basename(os.path.normpath(result_dir))
        target_dir = os.path.join(root_dir, test_dir)
        assert os.path.exists(target_dir), "Test dir {} not found in test data dir {}".format(test_dir, root_dir)
        files = os.listdir(target_dir)
        for f in files:
            result_file = os.path.join(result_dir, f)
            assert os.path.exists(result_file), "Test file {} not in result dir {}, test root {}".format(f, result_dir,
                                                                                                         root_dir)
            target_file = os.path.join(target_dir, f)
            status, msg = self.test_pickle(target_file, result_file)
            assert status, f"Test {test}/{subpath} failed, given {result_dir}, error: {msg}"
