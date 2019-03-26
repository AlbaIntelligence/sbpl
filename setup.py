from setuptools import setup, Extension, find_packages
import os
from distutils.sysconfig import get_config_vars


with open("README.md", "r") as f:
    long_description = f.read()

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

exec(open('sbpl/_version.py').read())

setup(
    name='sbpl',
    version=__version__,
    description='Python bindings for SBPL',
    long_description=long_description,
    author='Oleg Sinyavskiy',
    author_email='olegsinyavskiy@gmail.com',
    url='https://github.com/olegsinyavskiy/sbpl',
    download_url='',
    license='BSD License 2.0',
    install_requires=['numpy>=1.11.0'],
    package_data={'': ['input']},
    include_package_data=True,
    extras_require={
        'tests': ['pytest>=2.7.2',
                  'pytest-pep8>=1.0.6',
                  'pytest-xdist>=1.13.1',
                  'pytest-cov>=2.1.0'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages(),
    ext_modules=[Extension(
        'sbpl._sbpl_module',
        extra_compile_args=['-std=c++1y', '-O3'],
        include_dirs=['dep/pybind11/include',
                      'src/include'],
        sources=[
            'src/discrete_space_information/environment_nav2D.cpp',
            'src/discrete_space_information/environment_navxythetalat.cpp',
            'src/discrete_space_information/environment_navxythetamlevlat.cpp',
            'src/discrete_space_information/environment_nav2Duu.cpp',
            'src/discrete_space_information/environment_XXX.cpp',
            'src/discrete_space_information/environment_robarm.cpp',
            'src/heuristics/embedded_heuristic.cpp',
            'src/planners/adplanner.cpp',
            'src/planners/ANAplanner.cpp',
            'src/planners/araplanner.cpp',
            'src/planners/lazyARA.cpp',
            'src/planners/mhaplanner.cpp',
            'src/planners/ppcpplanner.cpp',
            'src/planners/rstarplanner.cpp',
            'src/planners/viplanner.cpp',
            'src/utils/heap.cpp',
            'src/utils/mdp.cpp',
            'src/utils/utils.cpp',
            'src/utils/2Dgridsearch.cpp',
            'src/utils/config.cpp',
            'src/python_wrapper.cpp'])
    ]
)
