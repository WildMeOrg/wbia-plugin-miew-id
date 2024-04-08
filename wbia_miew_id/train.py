import argparse
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import MiewIdDataset, get_train_transforms, get_valid_transforms
from logging_utils import WandbContext
from models import MiewIdNet
from etl import preprocess_data, print_intersect_stats, load_preprocessed_mapping, preprocess_dataset
from losses import fetch_loss
from schedulers import MiewIdScheduler
from engine import run_fn
from helpers import get_config, write_config
from torch.optim.swa_utils import AveragedModel, SWALR

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to the YAML configuration file. Default: configs/default_config.yaml')
    return parser.parse_args()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def set_seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run(rank, world_size, config):

    checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}"
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print('Checkpoints will be saved at: ', checkpoint_dir)

    config_path_out = f'{checkpoint_dir}/{config.exp_name}.yaml'
    config.data.test.checkpoint_path = f'{checkpoint_dir}/model_best.bin'

    def set_seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
    set_seed_torch(config.engine.seed)

    df_train = preprocess_data(config.data.train.anno_path, 
                                name_keys=config.data.name_keys,
                                convert_names_to_ids=True, 
                                viewpoint_list=config.data.viewpoint_list, 
                                n_filter_min=config.data.train.n_filter_min, 
                                n_subsample_max=config.data.train.n_subsample_max,
                                use_full_image_path=config.data.use_full_image_path,
                                images_dir = config.data.images_dir,
                                )

    df_val = preprocess_data(config.data.val.anno_path, 
                                name_keys=config.data.name_keys,
                                convert_names_to_ids=True, 
                                viewpoint_list=config.data.viewpoint_list, 
                                n_filter_min=config.data.val.n_filter_min, 
                                n_subsample_max=config.data.val.n_subsample_max,
                                use_full_image_path=config.data.use_full_image_path,
                                images_dir = config.data.images_dir
                                )
    
    print_intersect_stats(df_train, df_val, individual_key='name_orig')
    
    n_train_classes = df_train['name'].nunique()

    crop_bbox = config.data.crop_bbox
    
    # if config.data.preprocess_images.force_apply:
    #     preprocess_dir_images = os.path.join(checkpoint_dir, 'images')
    #     preprocess_dir_train = os.path.join(preprocess_dir_images, 'train')
    #     preprocess_dir_val = os.path.join(preprocess_dir_images, 'val')
    #     print("Preprocessing images. Destination: ", preprocess_dir_images)
    #     os.makedirs(preprocess_dir_train)
    #     os.makedirs(preprocess_dir_val)

    #     target_size = (config.data.image_size[0],config.data.image_size[1])

    #     df_train = preprocess_images(df_train, crop_bbox, preprocess_dir_train, target_size)
    #     df_val = preprocess_images(df_val, crop_bbox, preprocess_dir_val, target_size)

    #     crop_bbox = False

    if config.data.preprocess_images.apply:

        if config.data.preprocess_images.preprocessed_dir is None:
            preprocess_dir_images = os.path.join(checkpoint_dir, 'images')
        else:
            preprocess_dir_images = config.data.preprocess_images.preprocessed_dir

        if os.path.exists(preprocess_dir_images) and not config.data.preprocess_images.force_apply:
            print('Preprocessed images directory found at: ', preprocess_dir_images)
        else:
            preprocess_dataset(config, preprocess_dir_images)

        df_train = load_preprocessed_mapping(df_train, preprocess_dir_images)
        df_val = load_preprocessed_mapping(df_val, preprocess_dir_images)

        crop_bbox = False
    setup(rank, world_size)
    
    set_seed_torch(config.engine.seed)
    
    # Assume preprocess_data and other etl functions are compatible with DDP or are not necessary to run in each process
    # Make sure preprocess_data and similar functions are called before spawning processes if they're not compatible with DDP
    
    # Dataset preparation
    train_dataset = MiewIdDataset(
        csv=df_train,
        transforms=get_train_transforms(config),
        fliplr=config.test.fliplr,
        fliplr_view=config.test.fliplr_view,
        crop_bbox=crop_bbox,
    )
        
    valid_dataset = MiewIdDataset(
        csv=df_val,
        transforms=get_valid_transforms(config),
        fliplr=config.test.fliplr,
        fliplr_view=config.test.fliplr_view,
        crop_bbox=crop_bbox,
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.engine.train_batch_size, sampler=train_sampler,
        num_workers=config.engine.num_workers, pin_memory=True, drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.engine.valid_batch_size, sampler=valid_sampler,
        num_workers=config.engine.num_workers, pin_memory=True, drop_last=False,
    )

    # Model setup
    model = MiewIdNet(**dict(config.model_params))
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Loss, optimizer, scheduler, and SWA setup
    criterion = fetch_loss().to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.scheduler_params.lr_start)
    scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))
    if config.engine.use_swa:
        swa_model = AveragedModel(model.module) if isinstance(model, DDP) else AveragedModel(model)
        swa_model.to(rank)
        swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
        swa_start = config.swa_params.swa_start
    else:
        swa_model = None
        swa_scheduler = None
        swa_start = None

    # Training loop
    if rank == 0:
        with WandbContext(config):
            best_score = run_fn(rank, config, model, train_loader, valid_loader, criterion, optimizer, scheduler, rank, checkpoint_dir,
                                use_wandb=config.engine.use_wandb, swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)
    else:
        best_score = run_fn(rank, config, model, train_loader, valid_loader, criterion, optimizer, scheduler, rank, checkpoint_dir,
                            use_wandb=False, swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)
    
    cleanup()

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    config = get_config(config_path)

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(run, args=(world_size, config,), nprocs=world_size, join=True)
