from distutils.core import setup


setup(
    name='cort',
    version='0.1.5.1',
    packages=['cort',
              'cort.analysis',
              'cort.core',
              'cort.test',
              'cort.coreference',
              'cort.test.multigraph',
              'cort.test.analysis',
              'cort.test.core',
              'cort.coreference.multigraph',
              'cort.coreference.approaches',
              'cort.util'],

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
    install_requires=['nltk', 'numpy', 'matplotlib', 'mmh3', 'future'],
    package_data={
        'cort': ['analysis/visualization/style.css',
                 'analysis/visualization/lib/*',
                 'resources/*']
    },
    scripts=['bin/cort-train', 'bin/cort-predict', 'bin/run-multigraph'],
)
