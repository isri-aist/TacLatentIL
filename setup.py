from distutils.util import convert_path

from setuptools import setup

# https://stackoverflow.com/a/24517154
main_ns = {}
version_path = convert_path("tac_latent_il/version.py")
with open(version_path) as version_file:
    exec(version_file.read(), main_ns)

setup(
    name="tac_latent_il",
    version=main_ns["__version__"],
)
