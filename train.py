import argparse
import json
import os
import logging

import nni
from nni.utils import merge_parameter

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainSegmentationDataset as Dataset
from logger import Logger
from loss import DiceLoss
from transform import transforms
from unet import UNet
from utils import log_images, dsc

import nni.nas

import nni.nas.strategy as strategy
from nni.nas.experiment import NasExperiment
from nni.nas.evaluator import FunctionalEvaluator


#model_space = UNet()

if os.path.exists('log.log'):
    os.remove('log.log')

if os.path.exists('model.log'):
    os.remove('model.log')
    
result_logger = logging.getLogger('infomation')
result_logger.setLevel(logging.DEBUG)


# 创建文件处理器，并设置日志级别和文件名
file_handler = logging.FileHandler('log.log')
file_handler.setLevel(logging.DEBUG)

# 将文件处理器添加到 logger 对象中
result_logger.addHandler(file_handler)


# 创建另一个 logger 对象，用于记录 model_dict 的信息到另一个文件
model_logger = logging.getLogger('model infomation')
model_logger.setLevel(logging.INFO)

# 创建文件处理器，并设置日志级别和文件
model_file_handler = logging.FileHandler('model.log')
model_file_handler.setLevel(logging.INFO)

# 将文件处理器添加到 logger 对象中
model_logger.addHandler(model_file_handler)

#model_logger.info(model_space)

def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy

def evaluate_model(model):
    # By v3.0, the model will be instantiated by default.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)

    for epoch in range(5):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)

def main(args):

    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    makedirs(args)
    snapshotargs(args)
    
    
    # 实例化UNet，传递数据集的通道数
    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    unet.to(device)
    
    logger = Logger(args['logs'])
    result_logger.info(unet)
    result_logger.info(device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}



    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=args['lr'])
    loss_train = []
    loss_valid = []

    step = 0

    for epoch in tqdm(range(args['epochs']), total=args['epochs']):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        if (epoch % args['vis_freq'] == 0) or (epoch == args['epochs'] - 1):
                            if i * args['batch_size'] < args['vis_images']:
                                tag = "image/{}".format(i)
                                num_images = args['vis_images'] - i * args['batch_size']
                                logger.image_list_summary(
                                    tag,
                                    log_images(x, y_true, y_pred)[:num_images],
                                    step,
                                )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, prefix="val_")
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                        loader_valid.dataset.patient_slice_index,
                    )
                )
                logger.scalar_summary("val_dsc", mean_dsc, step)
                nni.report_intermediate_result(mean_dsc)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(args['weights'], "unet.pt"))
                loss_valid = []
    nni.report_final_result(best_validation_dsc)
    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=args['workers'],
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args['batch_size'],
        drop_last=False,
        num_workers=args['workers'],
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        images_dir=args['images'],
        subset="train",
        image_size=args['image_size'],
        transform=transforms(scale=args['aug_scale'], angle=args['aug_angle'], flip_prob=0.5),
    )
    valid = Dataset(
        images_dir=args['images'],
        subset="validation",
        image_size=args['image_size'],
        random_sampling=False,
    )
    return train, valid


def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args['weights'], exist_ok=True)
    os.makedirs(args['logs'], exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args['logs'], "args.json")
    with open(args_file, "w") as fp:
        json.dump(args, fp)

def get_params():
    parser = argparse.ArgumentParser(description='Training model for segmentation of brain MRI')

    parser.add_argument("--batch_size", type=int, default=4, metavar='N', help='input batch size for training (default: 4)')
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N', help='hidden layer size (default: 512)')
    parser.add_argument("--lr", type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument("--momentum", type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument("--epochs", type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument("--init_features", type=int, default=32, metavar='M', help='features for unet model (default: 32)')

    parser.add_argument("--seed", type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--no_cuda", action='store_true', default=False, help='disables CUDA training')
    parser.add_argument("--log_interval", type=int, default=1000, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument("--device",type=str,default="cuda:0",help="device for training (default: cuda:0)")

    parser.add_argument("--workers",type=int,default=4, help="number of workers for data loading (default: 4)")
    parser.add_argument("--vis-images",type=int,default=300, help="number of visualization images to save in log file (default: 200)")
    parser.add_argument("--vis-freq",type=int,default=10, help="frequency of saving images to log file (default: 10)")
    parser.add_argument("--weights", type=str, default="./weights", help="folder to save weights")
    parser.add_argument("--logs", type=str, default="./logs", help="folder to save logs")
    parser.add_argument( "--images", type=str, default="./data/kaggle_3m", help="root folder with images")
    parser.add_argument("--image-size",type=int,default=256,help="target input image size (default: 256)")
    parser.add_argument("--aug-scale",type=int,default=0.05,help="scale factor range for augmentation (default: 0.05)")
    parser.add_argument("--aug-angle",type=int,default=15,help="rotation angle range in degrees for augmentation (default: 15)")

    args, _ = parser.parse_known_args()
    return args



if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter()
        result_logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        result_logger.exception(exception)
        raise