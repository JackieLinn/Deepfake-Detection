import argparse
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.alexnet import AlexNet
from models.vgg import VGG19
from models.resnext import resnext101_32x8d as resnext101


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='alexnet',
                        choices=['alexnet', 'vgg', 'resnext'], help='model name')
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()
    print(f"Using model: {args.model}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "./figures/predict/0_big (1601).png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    if args.model == "alexnet":
        model = AlexNet(num_classes=args.num_classes).to(device)
    elif args.model == 'vgg':
        model = VGG19(num_classes=args.num_classes).to(device)
    elif args.model == 'resnext':
        model = resnext101(num_classes=args.num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}. Please select an existing model.")

    # load model weights
    weights_path = f"./save/best_{args.model}_model.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
