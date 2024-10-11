from setuptools import setup
import os

def get_requirements(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as req_file:
            return req_file.read().splitlines()
    return []

setup(
    install_requires=get_requirements('requirements.txt')
)
