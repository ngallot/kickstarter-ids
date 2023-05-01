import setuptools
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('kids/__init__.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)


setuptools.setup(
    name='kids',
    version=main_ns['__version__'],
    author='nicolas.gallot@gmail.com',
    description='Kickstarter campaing success predictor - an ml algorithm predicting the success of a kickstarter '
                'campaign',
    packages=setuptools.find_packages(),
    install_requires=[
        "pyspark==2.4.5",
        "click==7.0",
        "numpy==1.18.1",
        "mlflow==2.3.1",
        "google-cloud-storage==1.25.0"
    ],
    entry_points='''
    [console_scripts]
    kids=kids.cli:kids
    '''
)
