import argparse
import os

import nni
from nni.nas.pytorch import enas
from nni.utils import merge_parameter

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from medpy.filter.binary import largest_connected_component
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainSegmentationDataset as Dataset
from unet import UNet
from utils import dsc, gray2rgb, outline


def main(args):

    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    torch.manual_seed(args['seed'])
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args['device'])

    loader = data_loader(args)

    with torch.set_grad_enabled(False):
        # 定义搜索空间
        search_space = {
            'init_features': {'_type': 'choice', '_value': [16, 32, 64]},
        }

        # 实例化UNet，传递数据集的通道数
        unet0 = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)

        # 使用NNI中的ENAS接口进行搜索
        unet = enas.EnasUNet(unet0, search_space)
        unet.to(device)

        state_dict = torch.load(args['weights'], map_location=device)
        unet.load_state_dict(state_dict)
        unet.eval()
        unet.to(device)

        input_list = []
        pred_list = []
        true_list = []

        for i, data in tqdm(enumerate(loader)):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = unet(x)
            y_pred_np = y_pred.detach().cpu().numpy()
            pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

            y_true_np = y_true.detach().cpu().numpy()
            true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

            x_np = x.detach().cpu().numpy()
            input_list.extend([x_np[s] for s in range(x_np.shape[0])])

    volumes = postprocess_per_volume(
        input_list,
        pred_list,
        true_list,
        loader.dataset.patient_slice_index,
        loader.dataset.patients,
    )

    dsc_dist = dsc_distribution(volumes)

    dsc_dist_plot = plot_dsc(dsc_dist)
    imsave(args['figure'], dsc_dist_plot)

    for p in volumes:
        x = volumes[p][0]
        y_pred = volumes[p][1]
        y_true = volumes[p][2]
        for s in range(x.shape[0]):
            image = gray2rgb(x[s, 1])  # channel 1 is for FLAIR
            image = outline(image, y_pred[s, 0], color=[255, 0, 0])
            image = outline(image, y_true[s, 0], color=[0, 255, 0])
            filename = "{}-{}.png".format(p, str(s).zfill(2))
            filepath = os.path.join(args['predictions'], filename)
            imsave(filepath, image)


def data_loader(args):
    dataset = Dataset(
        images_dir=args['images'],
        subset="validation",
        image_size=args['image_size'],
        random_sampling=False,
    )
    loader = DataLoader(
        dataset, batch_size=args['batch_size'], drop_last=False, num_workers=1
    )
    return loader


def postprocess_per_volume(
    input_list, pred_list, true_list, patient_slice_index, patients
):
    volumes = {}
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        volume_in = np.array(input_list[index : index + num_slices[p]])
        volume_pred = np.round(
            np.array(pred_list[index : index + num_slices[p]])
        ).astype(int)
        volume_pred = largest_connected_component(volume_pred)
        volume_true = np.array(true_list[index : index + num_slices[p]])
        volumes[patients[p]] = (volume_in, volume_pred, volume_true)
        index += num_slices[p]
    return volumes


def dsc_distribution(volumes):
    dsc_dict = {}
    for p in volumes:
        y_pred = volumes[p][1]
        y_true = volumes[p][2]
        dsc_dict[p] = dsc(y_pred, y_true, lcc=False)
    return dsc_dict


def plot_dsc(dsc_dist):
    y_positions = np.arange(len(dsc_dist))
    dsc_dist = sorted(dsc_dist.items(), key=lambda x: x[1])
    values = [x[1] for x in dsc_dist]
    labels = [x[0] for x in dsc_dist]
    labels = ["_".join(l.split("_")[1:-1]) for l in labels]
    fig = plt.figure(figsize=(12, 8))
    canvas = FigureCanvasAgg(fig)
    plt.barh(y_positions, values, align="center", color="skyblue")
    plt.yticks(y_positions, labels)
    plt.xticks(np.arange(0.0, 1.0, 0.1))
    plt.xlim([0.0, 1.0])
    plt.gca().axvline(np.mean(values), color="tomato", linewidth=2)
    plt.gca().axvline(np.median(values), color="forestgreen", linewidth=2)
    plt.xlabel("Dice coefficient", fontsize="x-large")
    plt.gca().xaxis.grid(color="silver", alpha=0.5, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))


def makedirs(args):
    os.makedirs(args['predictions'], exist_ok=True)


def get_params():
    parser = argparse.ArgumentParser(description='Training model for segmentation of brain MRI')

    parser.add_argument("--batch_size", type=int, default=4, metavar='N', help='input batch size for training (default: 4)')
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N', help='hidden layer size (default: 512)')
    parser.add_argument("--lr", type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument("--momentum", type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument("--epochs", type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')

    parser.add_argument("--seed", type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--no_cuda", action='store_true', default=False, help='disables CUDA training')
    parser.add_argument("--log_interval", type=int, default=1000, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument("--device",type=str,default="cuda:0",help="device for training (default: cuda:0)")

    parser.add_argument("--workers",type=int,default=4, help="number of workers for data loading (default: 4)")
    parser.add_argument("--vis-images",type=int,default=200, help="number of visualization images to save in log file (default: 200)")
    parser.add_argument("--vis-freq",type=int,default=10, help="frequency of saving images to log file (default: 10)")
    parser.add_argument("--weights", type=str, default="./weights", help="folder to save weights")
    parser.add_argument("--logs", type=str, default="./logs", help="folder to save logs")
    parser.add_argument( "--images", type=str, default="./data_segentation", help="root folder with images")
    parser.add_argument("--image-size",type=int,default=256,help="target input image size (default: 256)")
    parser.add_argument("--aug-scale",type=int,default=0.05,help="scale factor range for augmentation (default: 0.05)")
    parser.add_argument("--aug-angle",type=int,default=15,help="rotation angle range in degrees for augmentation (default: 15)")
    parser.add_argument("--predictions",type=str,default="./predictions",help="folder for saving images with prediction outlines")
    parser.add_argument("--figure",type=str,default="./dsc.png",help="filename for DSC distribution figure")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
