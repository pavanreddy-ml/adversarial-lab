from setuptools import setup, find_packages

def get_requirements(filename):
    with open(filename, 'r') as req_file:
        return req_file.read().splitlines()

setup(
    name="adversary",
    version="0.1.0-dev",
    author="Pavan Reddy",
    author_email="NA",
    description="A unified library for performing adversarial attacks on ML model to test their defense.",
    long_description=open('README.md').read(),
    url="https://github.com/pavanreddy-ml/adversary",
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=get_requirements('requirements.txt'),
    license="MIT"
)
