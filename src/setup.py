from distutils.core import setup, Extension
import numpy

brunel = Extension("brunel", ["simlifmodule.cpp"],
                   include_dirs=[
                       '/usr/local/include', '/usr/local/include/c++',
                       numpy.get_include()
                   ],
                   language="c++")
#                 extra_compile_args=["-Ofast -stdlib=libc++ -lconfig++ -lgsl"])

setup(name="PackageName", ext_modules=[brunel])
