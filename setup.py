from setuptools import setup, find_packages

setup(
    name="torso",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "onnx>=1.16.0",
        "onnxruntime>=1.18.0",
        "onnx2torch>=1.5.15",
        "mlflow>=2.13.0",
        "numpy>=1.26.0,<2.0",
        "opencv-python-headless>=4.9.0",
        "Pillow>=10.3.0",
        "albumentations>=1.4.0",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.0",
    ],
)
