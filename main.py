import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from task_gate_decoder import TaskGateDecoder
from utils import *
from aspp import DeepLabHead
from create_dataset import NYUv2
from torch.utils.data import Subset
from LibMTL import Trainer
from LibMTL.model import resnet_dilated
from LibMTL.model.adapter_swin_transformer import adapter_swin_t
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
import wandb
from TwoSampler import *
# from data_utils import *

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--train_bs', default=8, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=8, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--ulw', default=1.0, type=float, help='unsupervised loss weighting')
    parser.add_argument('--pld', default=10, type=int, help='percentage of labelled data 10,25,50,75,100')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()
def TIT_decoder_head(tasks):
    decoders = nn.ModuleDict()
    encoder_dims = [(96 * 2**i) for i in range(4)]

    decoder = TaskGateDecoder(input_size=(9,12),
                                encoder_dims=encoder_dims,
                                embed_dim=96,
                                tasks=tasks, task_embed_dim=16, task_ind_dim=64)
    
    decoders['all'] = decoder
    return decoders   
def get_task_dict(task):
    # Define the base task components
    task_components = {'segmentation': {'metrics':['mIoU', 'pixAcc'], 
                              'metrics_fn': SegMetric(),
                              'loss_fn': SegLoss(),
                              'weight': [1, 1]}, 
                 'depth': {'metrics':['abs_err', 'rel_err'], 
                           'metrics_fn': DepthMetric(),
                           'loss_fn': DepthLoss(),
                           'weight': [0, 0]},
                 'normal': {'metrics':['mean', 'median', '<11.25', '<22.5', '<30'], 
                            'metrics_fn': NormalMetric(),
                            'loss_fn': NormalLoss(),
                            'weight': [0, 0, 1, 1, 1]}}
    
    # Split the task string and construct the dictionary
    task_keys = task.split('_')
    replacement_dict = {'Seg': 'segmentation', 'Dep': 'depth', 'Nor': 'normal'}
    total_out_channels = {'segmentation': 14, 'depth': 1, 'normal': 3}
    task_keys = [replacement_dict[key] for key in task_keys]
    num_out_channels = {}
    for key in task_keys:
        num_out_channels[key] = total_out_channels[key]
    task_dict = {component: task_components[component] for component in task_keys}

    return task_dict, num_out_channels, task_keys
def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    nyuv2_test_set = NYUv2(root=params.dataset_path, mode='test', augmentation=False,task=params.task)
    labeled_dataset = NYUv2(root=params.dataset_path, mode='labeled', augmentation=True, pld=params.pld,task=params.task)
    unlabeled_dataset = NYUv2(root=params.dataset_path, mode='unlabeled', augmentation=False, pld=params.pld,task=params.task)
    print(len(labeled_dataset), len(unlabeled_dataset))
    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=params.train_bs, shuffle=True)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=16, shuffle=False)
    
    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=params.test_bs,
        shuffle=False,
        pin_memory=True)
    
    task_dict, num_out_channels, tasks = get_task_dict(params.task)
    
    # define encoder and decoders
    def encoder_class(): 
        if params.arch == 'TIT':
            return adapter_swin_t(pretrained=True, img_size=((288, 384)),tasks=tasks)
        else:
            return resnet_dilated('resnet50')
    if params.arch == 'TIT':
        decoders = TIT_decoder_head(tasks)
    else:
        decoders = nn.ModuleDict({task: DeepLabHead(2048, 
                                                num_out_channels[task]) for task in list(task_dict.keys())})
    
    class NYUtrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
            super(NYUtrainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting_method.__dict__[weighting], 
                                            architecture=architecture_method.__dict__[architecture], 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)

        def process_preds(self, preds):
            img_size = (288, 384)
            for task in self.task_name:
                preds[task] = F.interpolate(preds[task], img_size, mode='bilinear', align_corners=True)
            return preds
    
    NYUmodel = NYUtrainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=encoder_class, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          **kwargs)
    if params.mode == 'train':
        # NYUmodel.train(nyuv2_train_loader, nyuv2_test_loader, params.epochs)
        NYUmodel.train_sl(labeled_loader, nyuv2_test_loader, params.epochs,unlabeled_dataloader=unlabeled_loader,bs=params.train_bs, labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset, ulw=params.ulw)
    elif params.mode == 'test':
        NYUmodel.test(nyuv2_test_loader)
    else:
        raise ValueError
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    wandb.init(project='abc', config=params)
    config = wandb.config
    config.aug = params.aug
    config.train_mode = params.train_mode
    config.train_bs = params.train_bs
    config.test_bs = params.test_bs
    config.epochs = params.epochs
    main(params)
