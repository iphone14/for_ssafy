<h1 align="center">Welcome to cnn for ssafy 👋</h1>
<p>
  <a href="https://hits.seeyoufarm.com">
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhibuz-cnn%2Fhit-counter&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/>
  </a>
</p>

> vanilla cnn

## Prerequisites
Conda (Optional)

```sh
conda create -n vanilla-cnn python
conda activate vanilla-cnn
```

```sh
pip install -r requirements.txt
```

## Usage

```sh
python cnn.py (Train & Test with default options)

python cnn.py -h
usage: CNN [-h] [-c [1-10]] [-m {light,complex}] [-g {Adam,SGD,RMSprop}] [-e EPOCHS] [-b BATCHES]

optional arguments:
  -h, --help            show this help message and exit
  -c [1-10]             classes (default: 3)
  -m {light,complex}    sample model type (default:light)
  -g {Adam,SGD,RMSprop}
                        sample gradient type (default: RMSprop)
  -e EPOCHS             epochs (default: 60)
  -b BATCHES            batches (default: classes x 3)
```

## Directory structure

```
.
├── README.md
├── cnn.py               main - CNN 실행파일
├── model.py             layer들을 모아서 forward,backward 실행
├── model_templates.py   example models
├── requirements.txt
├── utils.py             mnist file load, matrix 변환
├── layer                hidden layer, input, output layer class 모음
│   ├── abs_layer.py     layer의 추상 class
│   ├── convolution.py
│   ├── dense.py
│   ├── flatten.py
│   ├── input.py
│   └── max_pooling.py
├── gradient             gradient 종류 모음
│   ├── abs_gradient.py  gradient의 추상 class
│   ├── adam.py
│   ├── creator.py
│   ├── rms_prop.py
│   └── sgd.py
├── drawer               matrix 값들을 이미지 저장 (Optional)
│   ├── arial.ttf
│   └── drawer.py
└── mnist
    ├── train
    │   ├── zero (28x28 100 images)
    │   │   ├── 1912.png
    │   │   ├── ...
    │   │   └── 708.png
    │   ├── one
    │   ├── two
    │   ├── ...
    │   └── nine
    └── test
        ├── zero (28x28 5 images)
        │   ├── 1912.png
        │   ├── ...
        │   └── 708.png
        ├── one
        ├── two
        ├── ...
        └── nine
```

## Author

👤 **sweetchild222**


## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_