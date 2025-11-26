<div align="center">
    <h1>Deepfake-Detection</h1>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="">&nbsp;&nbsp;&nbsp;
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="">
</div>

<br>

This is a midterm assignment for the Introduction to Artificial Intelligence course at YNU.

On the other hand, this can also serve as a repository for various common classification models. The code is fully
encapsulated and can be run with one click, including the entire process of the classification task.

[TOC]

## Get Started

```bash
git clone https://github.com/JackieLinn/Deepfake-Detection.git
cd ./Deepfake-Detection/
```

## Requirements

```bash
pip install torch==2.8.0
pip install torchvision==0.23.0
pip install pillow==11.3.0
pip install matplotlib==3.6.3
pip install seaborn==0.13.2
pip install scikit-learn==1.6.1
pip install tqdm==4.67.1
pip install numpy==1.24.4
pip install pandas==1.5.3
```

## Environments

You can install the environment by the following method:

```bash
pip install -r requirements.txt
```

## Preparation

First, you need to put your dataset into the folder in the following format:

```bash
/dataset/train and /dataset/test
```

Then you need to do some data preprocessing work according to your own dataset format, including the training set and
test set obtained from the split mentioned earlier.

Additionally, you will need to create a `class_indices.json` file, which contains the categories of the classification,
in the following format:

```json
{
  "0": "class 1",
  "1": "class 2",
  "i": "class i",
  "n": "class n"
}
```

## Run scripts

You can run the training code using the following command:

```bash
python main.py --model model_name
```

And the prediction code can be executed using the following script:

```bash
python predict.py --model model_name
```

### If you are interested in this project, feel free to fork and star!
