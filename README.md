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
```
- Install pytorch
```
#cuda11.7
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
#or cuda11.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

```
- Install k2 with cuda11.7
```
wget https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20230927+cuda11.7.torch2.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install ./k2-1.24.4.dev20230927+cuda11.7.torch2.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```
- or Install k2 with cuda11.8
```
wget https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20230927+cuda11.8.torch2.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install ./k2-1.24.4.dev20230927+cuda11.8.torch2.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
``` 
> [!NOTE]
> if you want to install other k2 version: you can visit:https://k2-fsa.github.io/k2/cuda.html 

- Finally, you  need to install other packages
`pip install -r requirements.txt`

