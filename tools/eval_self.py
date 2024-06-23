import argparse
import os
import sys
sys.path.insert(0, './')

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from spice.config import Config
from spice.data.build_dataset import build_dataset
from spice.model.sim2sem import Sim2Sem
from spice.utils.miscellaneous import mkdir, save_config
import numpy as np
from spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari
from matplotlib.pyplot import imsave
from PIL import Image
import matplotlib.pyplot as plt
import PIL
import pickle
from tqdm import tqdm
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/speech_data/eval.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "--weight",
    default="./self_model_stl10.pth.tar",
    metavar="FILE",
    help="path to weight file",
    type=str,
)
parser.add_argument(
    "--all",
    default=1,
    type=int,
)
parser.add_argument(
    "--proto",
    default=0,
    type=int,
)
parser.add_argument(
    "--embedding",
    default="./results/stl10/embedding/feas_moco_512_l2.npy",
    type=str,
)


def main():
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    cfg.model.pretrained = args.weight
    cfg.proto = args.proto
    cfg.embedding = args.embedding
    cfg.all = False
    if cfg.all:
        cfg.data_test.split = "train+test"
        cfg.data_test.all = True
    else:
        cfg.data_test.split = "test"
        cfg.data_test.all = False

    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)
        if cfg.proto:
            mkdir("{}/proto".format(output_dir))

    output_config_path = os.path.join(output_dir, 'config.py')
    save_config(cfg, output_config_path)

    if cfg.gpu is not None:
        print("Use GPU: {}".format(cfg.gpu))

    # create model
    model = Sim2Sem(**cfg.model)
    print(model)

    torch.cuda.set_device(cfg.gpu)
    model = model.cuda(cfg.gpu)

    state_dict = torch.load(cfg.model.pretrained)
    for k in list(state_dict.keys()):
        # Initialize the feature module with encoder_q of moco.
        if k.startswith('module.'):
            # remove prefix
            # state_dict["module.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
            state_dict["{}".format(k[len('module.'):])] = state_dict[k]

        # delete renamed or unused k
        del state_dict[k]
    model.load_state_dict(state_dict)

    # Load similarity model
    cudnn.benchmark = True

    # Data loading code
    dataset_val = build_dataset(cfg.data_test) #create a new dataset for this function
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=1)

    model.eval()

    num_heads = len(cfg.model.head.multi_heads)
    assert num_heads == 1
  
    final_embeddings = {}

    for images, labels, idx in tqdm(val_loader):
        images = images.to(cfg.gpu, non_blocking=True) #[Batch,channels,96,96] = [100,3,96,96]
        with torch.no_grad():
            embds = model(images, forward_type="feature_with_head") #returns the features before the classification model. [Batch,512]
        embds = embds[0].detach().cpu()
        for index,l in enumerate(labels):
            if l in final_embeddings.keys():
                final_embeddings[str(l)].append(embds[index,:])
            else:
                final_embeddings[str(l)] = [embds[index,:]]
    with open('/home/workspace/yoavellinson/unsupervised_learning/SPICE/outputs/spec_embeds.pickle','wb') as f:
        pickle.dump(final_embeddings,f,protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
