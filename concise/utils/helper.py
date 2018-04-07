# my helper functions
import os
import json
import numpy as np
import time
import random
import string
import six
# read and save the object


def write_json(data, path):

    class NumpyAwareJSONEncoder(json.JSONEncoder):

        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            return json.JSONEncoder.default(self, obj)

    output = json.dumps(data, cls=NumpyAwareJSONEncoder)
    with open(os.path.expanduser(path), "w") as f:
        f.write(output)


def read_json(path):
    with open(os.path.expanduser(path)) as json_file:
        json_data = json.load(json_file)
    return json_data


# backcompatible code
def merge_dicts(*dict_args):
    """
    http://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression

    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def cur_time_str():
    time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    return time_str


def random_string(length=10):
    s = string.digits + string.ascii_letters
    return ''.join(random.sample(s, length))


def generate_random_file_path(dir, create_dir=True):
    if create_dir:
        if not os.path.exists(dir):
            os.makedirs(dir)
    file_path = dir + "/concise_" + cur_time_str() + "_" + random_string(10) + ".json"
    return file_path


def dict_to_numpy_dict(obj_dict):
    """
    Convert a dictionary of lists into a dictionary of numpy arrays
    """
    return {key: np.asarray(value) if value is not None else None for key, value in obj_dict.items()}


def rec_dict_to_numpy_dict(obj_dict):
    """
    Same as dict_to_numpy_dict, but recursive
    """
    if type(obj_dict) == dict:
        return {key: rec_dict_to_numpy_dict(value) if value is not None else None for key, value in obj_dict.items()}
    elif obj_dict is None:
        return None
    else:
        return np.asarray(obj_dict)


def compare_numpy_dict(a, b, exact=True):
    """
    Compare two recursive numpy dictionaries
    """
    if type(a) != type(b) and type(a) != np.ndarray and type(b) != np.ndarray:
        return False
    # go through a dictionary
    if type(a) == dict and type(b) == dict:
        if not a.keys() == b.keys():
            return False
        for key in a.keys():
            res = compare_numpy_dict(a[key], b[key], exact)
            if res == False:
                print("false for key = ", key)
                return False
        return True

    # if type(a) == np.ndarray and type(b) == np.ndarray:
    if type(a) == np.ndarray or type(b) == np.ndarray:
        if exact:
            return (a == b).all()
        else:
            return np.testing.assert_almost_equal(a, b)

    if a is None and b is None:
        return True

    raise NotImplementedError

    # try:
    #     res = a == b
    # except ValueError:
    #     res = (a == b).all()
    # return res


def numpy_dict_to_list(a):
    if type(a) == dict:
        return {key: numpy_dict_to_list(value) if value is not None else None for key, value in a.items()}
    elif a is None:
        return None
    elif type(a) is np.ndarray:
        return list(a).list()


def get_from_module(identifier, module_params, ignore_case=True):
    if ignore_case:
        _module_params = dict()
        for key, value in six.iteritems(module_params):
            _module_params[key.lower()] = value
        _identifier = identifier.lower()
    else:
        _module_params = module_params
        _identifier = identifier
    item = _module_params.get(_identifier)
    if not item:
        raise ValueError('Invalid identifier "%s"!' % identifier)
    return item


def _to_string(fn_str):
    if isinstance(fn_str, str):
        return fn_str
    elif callable(fn_str):
        return fn_str.__name__
    else:
        raise ValueError("fn_str has to be callable or str")
