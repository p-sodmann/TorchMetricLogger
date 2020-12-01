from setuptools import setup, find_packages

print(f"Found packages: {find_packages()}")

setup(
    name='torchmetriclogger',
    packages=find_packages(),
    description='Useful to log metrics while training a pytorch model',
    version='0.1',
    url='',
    author='Philipp Sodmann',
    author_email='psodmann@gmail.com',
    keywords=['Pytorch', "Metrics"]
)
