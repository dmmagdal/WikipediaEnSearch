# setup.py
from setuptools import setup
# from setuptools.extension import Extension
from Cython.Build import cythonize

# extensions = [
# 	Extension("load_msgpack", ["preprocess_helper.pyx"], libraries=["msgpackc"]),
# ]

setup(
	# name="msgpack_loader",
	# ext_modules=cythonize(extensions),
	ext_modules=cythonize(
		# "search_helper.pyx", annotate=True
		"search.pyx", annotate=True
	),
)
