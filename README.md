# foundnet: Foundational Network for Pretraining Toolkit

# foundnet design principles
This repository follows the main principle of openness through clarity.
foundnet is:

- **Simple**: Clear code, no use of multi-level class inheritance.
- **Correct**: Numerically equivalent to the original model.
- **Modularisation**: Reduced dependencies between models for easy reading and understand of code.
- **Completeness**: Provides a complete recipe for preparing data, training and inference

# Install for training
- Clone the repo
`git clone https://github.com/shanguanma/foundnet.git`
- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:
```
conda create -n foundnet python=3.11 -y
conda activate foundnet
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
