import setuptools
from inside_analysis.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = list(f)

setuptools.setup(
                 name=u"inside_analysis",
                 version=__version__,
                 author=u"Greta Del Nista",
                 author_email="greta.delnista@gmail.com",
                 description="Toolkit to perform image analysis.",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://github.com/pypa/sampleproject",
                 packages=setuptools.find_packages(),
                 classifiers=[
                              "Programming Language :: Python :: 3",
                              "License :: GNU GENERAL PUBLIC LICENSE",
                              "Operating System :: OS Independent",
                              ],
                 python_requires='>=3.6',
                 install_requires = requirements
                 )
