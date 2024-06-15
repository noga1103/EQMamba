from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="EQTransformer",
    version="0.1.61",
    description="A python package for making and using attentive deep-learning models for earthquake signal detection and phase picking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smousavi05/EQTransformer",
    license="MIT",
    packages=find_packages(),
    keywords='Seismology, Earthquakes Detection, P&S Picking, Deep Learning, Attention Mechanism',
    install_requires=[
        'pytest',
        'numpy',
        'keyring',
        'pkginfo',
        'scipy',
        # 'tensorflow-deps',
        # 'tensorflow-estimator',
        # 'tensorflow-macos',
        'tensorflow',
        'keras',
        'matplotlib',
        'pandas',
        'tqdm',
        'h5py',
        'obspy',
        'jupyter'
    ],
   
)

