from distutils.core import setup, Extension
import numpy

setup(name             = "sequence_shuffle",
      version          = "1.0",
      description      = "Numpy C API template.",
      author           = "Michael Siebert",
      author_email     = "michael.siebert2k@gmail.com",
      maintainer       = "michael.siebert2k@gmail.com",
      ext_modules      = [
            Extension(
                  'ext.sequence_shuffle', ['src/sequence_shuffle.cpp'],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=[
                        "-Ofast",
                        "-march=native"
                  ]
            ),
      ],

)