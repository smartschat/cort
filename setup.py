from distutils.core import setup
import setuptools

setup(
    name='cort',
    version='0.1.2',
    packages=setuptools.find_packages(),
    url='http://github.com/smartschat/cort',
    license='MIT',
    author='Sebastian Martschat, Thierry Goeckel',
    author_email='sebastian.martschat@gmail.com',
    description='A coreference resolution research toolkit.',
    keywords = ['NLP', 'CL', 'natural language processing',
                'computational linguistics', 'coreference resolution',
                'text analytics'],
    classifiers = [
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
        ],
    install_requires=['nltk', 'numpy', 'matplotlib', 'mmh3'],
    package_data={
        'visualization': 'cort/analysis/visualization',
        'resources': 'cort/resources'
    },
    scripts=['bin/cort-train', 'bin/cort-predict', 'bin/run-multigraph']
)
