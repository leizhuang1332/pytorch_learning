from setuptools import setup, find_packages

setup(
    # 其他配置...
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
