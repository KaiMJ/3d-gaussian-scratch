from setuptools import setup

setup(
    packages=["cuda_backend"],
    package_dir={"": "src"},
    package_data={
        'cuda_backend': ['*.pyd', '*.so']
    },
)