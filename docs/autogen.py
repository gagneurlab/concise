# -*- coding: utf-8 -*-
'''
General documentation architecture:

Home
Index

- Getting started
    Getting started to concise

Layers
- Preprocessing
    Genomic Sequence Preprocessing
    RNA Structure Preprocessing
    Spline-position Preprocessing

- Data
    Encode
    Attract

Losses
Metrics
Eval metrics
Optimizers
Activations
Initializers
Regularizers
- Utils
    fasta
    model_data
    pwm
    splines
Contributing

'''
from __future__ import print_function
from __future__ import unicode_literals

import re
import inspect
import os
import shutil
import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding('utf8')

import concise
from concise import utils
from concise.utils import fasta
from concise.utils import helper
from concise.utils import model_data
from concise.utils import pwm
from concise.utils import splines
from concise.data import encode
from concise.data import attract
from concise.preprocessing import sequence
from concise.preprocessing import smooth
from concise.preprocessing import structure
from concise import activations
from concise import constraints
from concise import eval_metrics
from concise import metrics
from concise import hyopt
from concise import initializers
from concise import layers
from concise import losses
from concise import optimizers
from concise import regularizers


EXCLUDE = {
    'Optimizer',
    'Wrapper',
    'get_session',
    'set_session',
    'CallbackList',
    'serialize',
    'deserialize',
    'get',
}

PAGES = [
    {
        'page': 'preprocessing/sequence.md',
        'functions': [
            sequence.encodeDNA,
            sequence.encodeRNA,
            sequence.encodeCodon,
            sequence.encodeAA,
            sequence.pad_sequences,
            sequence.encodeSequence,
        ]
    },
    {
        'page': 'preprocessing/splines.md',
        'functions': [
            smooth.encodeSplines,
        ]
    },
    {
        'page': 'preprocessing/structure.md',
        'functions': [
            structure.encodeRNAStructure,
        ]
    },
    {
        'page': 'layers.md',
        'functions': [
            layers.InputDNA,
            layers.InputRNA,
            layers.InputRNAStructure,
            layers.InputCodon,
            layers.InputAA,
            layers.InputSplines,
        ],
        'classes': [
            layers.ConvDNA,
            layers.ConvRNA,
            layers.ConvRNAStructure,
            layers.ConvAA,
            layers.ConvCodon,
            layers.ConvSplines,
            layers.SmoothPositionWeight,
            layers.GlobalSumPooling1D,
        ],
    },
    {
        'page': 'losses.md',
        'all_module_functions': [losses],
    },
    {
        'page': 'metrics.md',
        'all_module_functions': [metrics],
    },
    {
        'page': 'eval_metrics.md',
        'all_module_functions': [eval_metrics],
    },
    {
        'page': 'initializers.md',
        'all_module_functions': [initializers],
        'all_module_classes': [initializers],
    },
    {
        'page': 'regularizers.md',
        'all_module_functions': [regularizers],
        'all_module_classes': [regularizers],
    },
    {
        'page': 'optimizers.md',
        'all_module_classes': [optimizers],
        'functions': [
            optimizers.data_based_init
        ]
    },
    {
        'page': 'activations.md',
        'all_module_functions': [activations],
    },
    {
        'page': 'utils/fasta.md',
        'all_module_functions': [utils.fasta],
    },
    {
        'page': 'utils/model_data.md',
        'all_module_functions': [utils.model_data],
    },
    {
        'page': 'utils/pwm.md',
        'classes': [utils.pwm.PWM]
    },
    {
        'page': 'utils/splines.md',
        'classes': [utils.splines.BSpline]
    },
    {
        'page': 'hyopt.md',
        'classes': [
            hyopt.CMongoTrials,
            hyopt.CompileFN,
        ],
        'functions': [
            hyopt.test_fn,
            hyopt.eval_model,
        ]
    },
    {
        'page': 'data/encode.md',
        'functions': [
            encode.get_metadata,
            encode.get_pwm_list,
        ]
    },
    {
        'page': 'data/attract.md',
        'functions': [
            attract.get_metadata,
            attract.get_pwm_list,
        ]
    },
]

# TODO
ROOT = 'http://concise.io/'


def get_earliest_class_that_defined_member(member, cls):
    ancestors = get_classes_ancestors([cls])
    result = None
    for ancestor in ancestors:
        if member in dir(ancestor):
            result = ancestor
    if not result:
        return cls
    return result


def get_classes_ancestors(classes):
    ancestors = []
    for cls in classes:
        ancestors += cls.__bases__
    filtered_ancestors = []
    for ancestor in ancestors:
        if ancestor.__name__ in ['object']:
            continue
        filtered_ancestors.append(ancestor)
    if filtered_ancestors:
        return filtered_ancestors + get_classes_ancestors(filtered_ancestors)
    else:
        return filtered_ancestors


def get_function_signature(function, method=True):
    signature = getattr(function, '_legacy_support_signature', None)
    if signature is None:
        signature = inspect.getargspec(function)
    defaults = signature.defaults
    if method:
        args = signature.args[1:]
    else:
        args = signature.args
    if defaults:
        kwargs = zip(args[-len(defaults):], defaults)
        args = args[:-len(defaults)]
    else:
        kwargs = []
    st = '%s.%s(' % (function.__module__, function.__name__)
    for a in args:
        st += str(a) + ', '
    for a, v in kwargs:
        if isinstance(v, str):
            v = '\'' + v + '\''
        st += str(a) + '=' + str(v) + ', '
    if kwargs or args:
        return st[:-2] + ')'
    else:
        return st + ')'


def get_class_signature(cls):
    try:
        class_signature = get_function_signature(cls.__init__)
        class_signature = class_signature.replace('__init__', cls.__name__)
    except:
        # in case the class inherits from object and does not
        # define __init__
        class_signature = cls.__module__ + '.' + cls.__name__ + '()'
    return class_signature


def class_to_docs_link(cls):
    module_name = cls.__module__
    assert module_name[:8] == 'concise.'
    module_name = module_name[8:]
    link = ROOT + module_name.replace('.', '/') + '#' + cls.__name__.lower()
    return link


def class_to_source_link(cls):
    module_name = cls.__module__
    assert module_name[:8] == 'concise.'
    path = module_name.replace('.', '/')
    path += '.py'
    line = inspect.getsourcelines(cls)[-1]
    link = 'https://github.com/avsecz/concise/blob/master/' + path + '#L' + str(line)
    return '[[source]](' + link + ')'


def code_snippet(snippet):
    result = '```python\n'
    result += snippet + '\n'
    result += '```\n'
    return result


def process_class_docstring(docstring):
    docstring = re.sub(r'\n    # (.*)\n',
                       r'\n    __\1__\n\n',
                       docstring)

    docstring = re.sub(r'    ([^\s\\]+):(.*)\n',
                       r'    - __\1__:\2\n',
                       docstring)

    docstring = docstring.replace('    ' * 5, '\t\t')
    docstring = docstring.replace('    ' * 3, '\t')
    docstring = docstring.replace('    ', '')
    return docstring


def process_function_docstring(docstring):
    docstring = re.sub(r'\n    # (.*)\n',
                       r'\n    __\1__\n\n',
                       docstring)
    docstring = re.sub(r'\n        # (.*)\n',
                       r'\n        __\1__\n\n',
                       docstring)

    docstring = re.sub(r'    ([^\s\\]+):(.*)\n',
                       r'    - __\1__:\2\n',
                       docstring)

    docstring = docstring.replace('    ' * 6, '\t\t')
    docstring = docstring.replace('    ' * 4, '\t')
    docstring = docstring.replace('    ', '')
    return docstring


print('Cleaning up existing sources directory.')
if os.path.exists('sources'):
    shutil.rmtree('sources')

print('Populating sources directory with templates.')
for subdir, dirs, fnames in os.walk('templates'):
    for fname in fnames:
        new_subdir = subdir.replace('templates', 'sources')
        if not os.path.exists(new_subdir):
            os.makedirs(new_subdir)
        if fname[-3:] == '.md':
            fpath = os.path.join(subdir, fname)
            new_fpath = fpath.replace('templates', 'sources')
            shutil.copy(fpath, new_fpath)

# Take care of index page.
readme = open('../README.md').read()
index = open('templates/index.md').read()
index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
f = open('sources/index.md', 'w')
f.write(index)
f.close()

print('Starting autogeneration.')
for page_data in PAGES:
    blocks = []
    classes = page_data.get('classes', [])
    for module in page_data.get('all_module_classes', []):
        module_classes = []
        for name in dir(module):
            if name[0] == '_' or name in EXCLUDE:
                continue
            module_member = getattr(module, name)
            if inspect.isclass(module_member):
                cls = module_member
                if cls.__module__ == module.__name__:
                    if cls not in module_classes:
                        module_classes.append(cls)
        module_classes.sort(key=lambda x: id(x))
        classes += module_classes

    for cls in classes:
        subblocks = []
        signature = get_class_signature(cls)
        subblocks.append('<span style="float:right;">' + class_to_source_link(cls) + '</span>')
        subblocks.append('### ' + cls.__name__ + '\n')
        subblocks.append(code_snippet(signature))
        docstring = cls.__doc__
        if docstring:
            subblocks.append(process_class_docstring(docstring))
        blocks.append('\n'.join(subblocks))

    functions = page_data.get('functions', [])
    for module in page_data.get('all_module_functions', []):
        module_functions = []
        for name in dir(module):
            if name[0] == '_' or name in EXCLUDE:
                continue
            module_member = getattr(module, name)
            if inspect.isfunction(module_member):
                function = module_member
                if module.__name__ in function.__module__:
                    if function not in module_functions:
                        module_functions.append(function)
        module_functions.sort(key=lambda x: id(x))
        functions += module_functions

    for function in functions:
        subblocks = []
        signature = get_function_signature(function, method=False)
        signature = signature.replace(function.__module__ + '.', '')
        subblocks.append('### ' + function.__name__ + '\n')
        subblocks.append(code_snippet(signature))
        docstring = function.__doc__
        if docstring:
            subblocks.append(process_function_docstring(docstring))
        blocks.append('\n\n'.join(subblocks))

    if not blocks:
        raise RuntimeError('Found no content for page ' +
                           page_data['page'])

    mkdown = '\n----\n\n'.join(blocks)
    # save module page.
    # Either insert content into existing page,
    # or create page otherwise
    page_name = page_data['page']
    path = os.path.join('sources', page_name)
    if os.path.exists(path):
        template = open(path).read()
        assert '{{autogenerated}}' in template, ('Template found for ' + path +
                                                 ' but missing {{autogenerated}} tag.')
        mkdown = template.replace('{{autogenerated}}', mkdown)
        print('...inserting autogenerated content into template:', path)
    else:
        print('...creating new page with autogenerated content:', path)
    subdir = os.path.dirname(path)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    open(path, 'w').write(mkdown)

shutil.copyfile('../CONTRIBUTING.md', 'sources/contributing.md')
