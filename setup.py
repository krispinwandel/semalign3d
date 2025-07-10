from setuptools import setup, find_packages


setup(
    name="semalign3d",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core ML/DL frameworks
        "torch==2.1.2",
        "torchvision==0.16.2",
        "cudf-cu12==24.8.3",
        "cuml-cu12==24.8.0",
        "cupy-cuda12x==13.3.0",
        "fvcore",
        "pyquaternion",
        "transformers",
        "einops==0.8.0",
        "segment_anything @ git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588#egg=segment_anything",
        # Scientific computing
        "numpy==1.26.3",
        "scipy==1.13.0",
        "pandas==2.2.0",
        # Computer vision and image processing
        "opencv-python==4.9.0.80",
        "Pillow==10.0.0",
        # Progress bars and UI
        "tqdm==4.66.4",
        "rich==13.7.0",  # for console panels
        # Configuration and serialization
        "PyYAML==6.0.1",
        # File I/O and data formats
        "h5py==3.11.0",
        # TODO adjust torch version so that we can install xformers more easily
        # Optional performance optimizations
        # "xformers==0.0.23",
        # Visualization
        "plotly==5.22.0",
        "jupyterlab==4.1.8",
        "jupyterlab_widgets==3.0.10",
        "ipywidgets>=7.0.0",
    ],
    python_requires=">=3.10",
    author="Krispin Wandel",
    description="SemAlign3D: Semantic Correspondence between RGB-Images through Aligning 3D Object-Class Representations",
    url="https://github.com/krispinwandel/semalign3d",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
