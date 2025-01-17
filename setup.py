from setuptools import find_packages, setup

# you should run setup.sh instead of this one, setup.sh will call this package
setup(
    name="codeRM_eval",
    version="1.0.0",
    description="",
    author="Wyett (Huaye) Zeng",
    author_email="wyettzeng@gmail.com",
    packages=find_packages(),
    url="https://github.com/wyettzeng/CodeDPO",  # github
    install_requires=[
        # comment the following if you have CUDA 11.8
        "pandas",
    ],
)
