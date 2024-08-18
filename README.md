# Multi-Task-Learning-with-Limited-Labels

This project aims to implement Multi-Task Learning models with limited labels.
## Requirements

To run this project, you will need the following dependencies:

*   python==3.8.19
*   numpy>=1.24.4
*   torch==2.1.2
*   torchvision==0.16.2
*   scipy==1.10.1
*   timm==1.0.7
*   wandb==0.16.6

## Setup
1. Create a conda environment using the following command:
    ```
      Conda env create -n abc python=3.8
    ```
2. Install PyTorch and Torchvision with the command:
    ```
      pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
    ```
3. Install other packages using the command:
    ```
      pip install scipy==1.10.1 timm==1.0.7 wandb==0.16.6
    ```
4.  Install packages using the setup.py by the command:
    ```
    pip install -e .
    ```
5. Copy and extract the dataset available at [NYU Depth V2](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) and this was processed by [mtan](https://github.com/lorenmt/mtan).

## Usage

`python main.py` executes and runs the code with all the default arguments.

You can also customize the settings by using the following command line arguments:

*   `--weighting`: Choose the weighting scheme.
*   `--arch`: Choose the architecture.
*   `--dataset_path`: Provide the dataset folder path.
*   `--gpu_id`: Choose between the available GPU's.
*   `--scheduler`: Choose between available schedulers.
*   `--epochs`: Provide the number of epochs to train.
*   `--save_path`: Provide the path to save the checkpoint file.
*   `--ulw`: Set the `lambda` value as float (if it is greater than 5, the semi-supervised learning with a warmup of 10 epochs is carried out).
*   `--pld`: Set the percentage of labeled data and the remaining is used as unlabeled data as semi-supervised training is carried out.
*   `--task`: Set the what task should be carried out between Segmentation (Seg), Depth (Dep), Normal (Nor) and combinations of two tasks or three tasks using multi-task learning.

## Acknowledgements

I want to thank the author who created the library for multi-task learning and the authors who released the public repositories: [LibMTL](https://github.com/median-research-group/LibMTL), [MTDP_Lib](https://github.com/innovator-zero/MTDP_Lib), [CAGrad](https://github.com/Cranial-XIX/CAGrad), [dselect_k_moe](https://github.com/google-research/google-research/tree/master/dselect_k_moe), [MultiObjectiveOptimization](https://github.com/isl-org/MultiObjectiveOptimization), [mtan](https://github.com/lorenmt/mtan), [MTL](https://github.com/SamsungLabs/MTL), [nash-mtl](https://github.com/AvivNavon/nash-mtl), [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric), and [xtreme](https://github.com/google-research/xtreme).
