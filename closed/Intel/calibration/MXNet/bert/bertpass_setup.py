"""
setup.py: prepares required libraries for BERT scripts
"""
#!/usr/bin/env python
import pathlib
import sys
import os
import logging
from distutils.command.install import install
from setuptools import setup
import mxnet

requirements = [
    'numpy>=1.16.0',
]

def CompileBERTCustomPass():
    """Compiles custom graph pass for BERT into a library. It offers performance improvements"""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log = logging.getLogger()
    input_pass_file = 'bertpass.cc'
    out_lib_file = 'bertpass.so'
    log.info(' ... compiling BERT custom graph pass into %s', out_lib_file)
    mxnet_path = pathlib.Path("/path/to/incubator-mxnet")
    mxnet_include_path = pathlib.Path.joinpath(mxnet_path, 'include')
    mxnet_lib_api_path = pathlib.Path.joinpath(mxnet_path, 'src', 'lib_api.cc')
    pass_path = os.path.dirname(os.path.realpath(__file__))
    source = os.path.join(pass_path, input_pass_file)
    target = os.path.join(pass_path, out_lib_file)
    os.system('g++ -shared -g -fPIC -std=c++11 ' + str(source) + ' ' + str(mxnet_lib_api_path) +
              ' -o ' + str(target) + ' -I ' + str(mxnet_include_path) + ' -DMSHADOW_USE_CUDA=0 -DMXNET_USE_CUDA=0 -DUSE_INTEL_PATH=/opt/intel/')

class CompileBERTPass(install):
    def run(self):
        install.run(self)
        self.execute(CompileBERTCustomPass, ())

setup(
    # Metadata
    name='gluonnlp-scripts-bert',
    python_requires='>=3.5',
    author='Gluon NLP Toolkit Contributors',
    author_email='mxnet-gluon@amazon.com',
    url='https://github.com/dmlc/gluon-nlp',
    description='MXNet Gluon NLP Toolkit - BERT scripts',
    license='Apache-2.0',
    cmdclass={'install': CompileBERTPass}
)
