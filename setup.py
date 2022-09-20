from setuptools import setup, find_packages

long_description = """SYCL code generation with a focus on particles based operations."""

install_requires = []
with open('requirements.txt') as fh:
    for l in fh:
        if len(l) > 0:
            install_requires.append(l)

setup(
   name='galle',
   version='1.0',
   description='NESO',
   license="GPL3",
   long_description=long_description,
   author='Will Saunders',
   author_email='will.saunders@ukaea.uk',
   url="",
   packages=find_packages(),
   install_requires=install_requires,
   scripts=[],
   include_package_data=True
)
