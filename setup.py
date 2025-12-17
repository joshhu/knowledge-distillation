from setuptools import setup, find_packages

setup(
    name="knowledge-distillation",
    version="0.1.0",
    description="知識蒸餾教學專案 - 使用教師模型訓練學生模型",
    author="Your Name",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
