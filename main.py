import argparse

from train import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='alexnet',
                        choices=['alexnet', 'googlenet', 'resnet', 'resnext', 'densenet', 'swint', 'mobilenet'],
                        help='model name')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--data-path', type=str, default="./dataset")
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    options = parser.parse_args()

    run(options)


if __name__ == '__main__':
    main()
