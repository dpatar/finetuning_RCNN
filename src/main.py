import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
from tqdm import tqdm
import logging

# Paths
IMG_DATA_PATH = r"D:\Datasets\exDark_dataset\ExDark"
MASK_DATA_PATH = r"D:\Datasets\exDark_dataset\ExDark_Annno"


class ExDarkDataset(Dataset):
    def __init__(self, img_root=IMG_DATA_PATH, mask_root=MASK_DATA_PATH, transforms=None):
        self.img_root = img_root
        self.mask_root = mask_root
        self.transforms = transforms
        # Loading paths images and masks
        self.imgs = []
        self.masks = []
        self.labels = dict(zip(os.listdir(img_root), range(1, len(os.listdir(img_root))+1)))
        for label in self.labels.keys():
            samples = os.listdir(os.path.join(img_root, label))
            for eachSample in samples:
                sample = os.path.join(os.path.join(img_root, label), eachSample)
                self.imgs.append(sample)
                sample_mask = os.path.join(os.path.join(mask_root, label), eachSample) + r".txt"
                self.masks.append(sample_mask)

    def __getitem__(self, idx):
        # Load imgs and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        img = torchvision.transforms.ToTensor()(img) #.unsqueeze_(0)
        # tensor_to_pil = transforms.ToPILImage()(pil_to_tensor.squeeze_(0))

        # getting each box for each object in an image
        maskFile = pd.read_csv(mask_path, header=None, delim_whitespace=True, skiprows=1)

        # Objects Label Name and Label_id from Hash_table
        objs_label_names = maskFile[maskFile.columns[0]].values.tolist()
        label_ids = []
        for key in objs_label_names:
            label_ids.append(self.labels.get(key))
        label_ids = torch.as_tensor(label_ids, dtype=torch.int64)
        # Bounding box coordinates [l t w h]
        # l - pixel number from left of image
        # t - pixel number from top of image
        # w - width of bounding box
        # h - height of bounding box
        boxes = maskFile[maskFile.columns[1:5]].values.tolist()
        adjusted_boxes = []
        for box_i in boxes:
            adjusted_box_i = [0, 0, 0, 0]
            adjusted_box_i[0] = box_i[0]
            adjusted_box_i[1] = box_i[1]
            adjusted_box_i[2] = box_i[0] + box_i[2]
            adjusted_box_i[3] = box_i[1] + box_i[3]
            adjusted_boxes.append(adjusted_box_i)

        boxes = torch.as_tensor(adjusted_boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(objs_label_names),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": label_ids, "area": area, "image_id": image_id, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_label_name(self, label_id):
        return list(self.labels)[label_id-1]


def dataset_split(dataset_presplit, ratio=0.8):
    trainSize = int(ratio * len(dataset_presplit))
    testSize = len(dataset_presplit) - trainSize
    set_seeds()
    return torch.utils.data.random_split(dataset_presplit, [trainSize, testSize])


def getDataloaders(dataset_to_loader, ratio=0.8, get_datasets=False):
    train_dataset, test_dataset = dataset_split(dataset_to_loader, ratio)
    if get_datasets:
        return train_dataset, test_dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                               collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn)
    return train_loader, test_loader


def get_RCNN_Model(number_classes=13):
    # load a model
    rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = rcnn_model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, number_classes)
    return rcnn_model


def collate_fn(batch):
    return tuple(zip(*batch))


def set_seeds():
    torch.manual_seed(0)
    # np.random.seed(0)

def logger_init():
    logging.basicConfig(filename=r'..\train.log', level=logging.INFO)


if __name__ == "__main__":
    logger_init()
    # Parameters
    train_test_ratio = 0.8
    batch_size = 1
    num_classes = 13
    lr_rate = 0.005
    weight_decay = 0.0005
    num_epochs = 20
    set_seeds()
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Dataset
    dataset = ExDarkDataset(IMG_DATA_PATH, MASK_DATA_PATH)
    train_data_loader, test_data_loader = getDataloaders(dataset, train_test_ratio)
    # Model
    model = get_RCNN_Model(num_classes)
    model = model.to(device)
    # Optimizer & Schedular
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Model Training
    model.train()
    print("Started Training")
    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        progress_bar = tqdm(train_data_loader, position=0, desc="Epoch {}".format(epoch))
        running_loss = 0
        count = 0

        for images, targets in progress_bar:
            # To GPU
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss = running_loss + losses
            count = count + 1

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Catch Your Breath
            time.sleep(0.005)

        lr_scheduler.step()
        # Epoch Print Info
        current_time = time.perf_counter()
        # Saving info to log file
        logging.info("Epoch {}/{} Done, Total Loss: {}"
              .format(epoch, num_epochs, running_loss / count))
        logging.info("Total Time Elapsed: {} seconds"
              .format(str(current_time - start_time)))

        # Save model of each epoch
        model_file_name = r"..\models\model_1_epoch_{}.pth".format(epoch)
        torch.save(model.state_dict(), model_file_name)

        # Testing Part
        with torch.no_grad():
            running_loss_test = 0
            count = 0
            test_bar = tqdm(test_data_loader, position=0)
            for images, targets in test_bar:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                running_loss_test = running_loss_test + losses
                count = count + 1
                time.sleep(0.005)
            print("Epoch {}, Test Total Loss: {}".format(epoch, running_loss_test / count))
            logging.info("Epoch {}, Test Total Loss: {}".format(epoch, running_loss_test / count))
