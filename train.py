import os

import torch
import torch.optim as optim
from torchvision import transforms

from my_dataset import MyDataSet
from utils import set_random_seed, read_data, train_loop, get_logger, create_model


def run(args):
    set_random_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {}".format(device))

    train_images_path, train_images_label, test_images_path, test_images_label = read_data(args.data_path)

    img_size = 256
    data_transform = {
        "train": transforms.Compose([
            # transforms.Resize(int(img_size * 1.143)),
            # transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(90),  # 90度倍数随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # 仿射变换
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
        ]),
        "test": transforms.Compose([
            # transforms.Resize(int(img_size * 1.143)),
            # transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform["test"])

    batch_size = args.batch_size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(num_workers))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=num_workers,
                                               collate_fn=train_dataset.collate_fn)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=test_dataset.collate_fn)

    # 构造模型
    model = create_model(args.model, args.device, args.num_classes)
    print(f"Using model: {args.model}")

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # 设置优化器
    parameter_group = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(parameter_group, lr=args.lr, weight_decay=5E-2)

    best_acc = 0.0
    logger = get_logger(args.model)
    logger.info("Start training and validating...")
    for epoch in range(args.epochs):
        test_acc, all_labels, all_preds = train_loop(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epoch=epoch,
            logger=logger,
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"./save/best_{args.model}_model.pth")
            logger.info("Best model saved!")

            # 保存 all_labels 和 all_preds 到 txt 文件
            with open(f"./logs/best_{args.model}_model.txt", "w") as f:
                for label, pred in zip(all_labels, all_preds):
                    f.write(f"{label} {pred}\n")
