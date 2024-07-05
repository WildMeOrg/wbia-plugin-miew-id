from wbia_miew_id.logging_utils import WandbContext
from wbia_miew_id.etl import preprocess_data, print_intersect_stats, load_preprocessed_mapping, preprocess_dataset
from wbia_miew_id.losses import fetch_loss
from wbia_miew_id.schedulers import MiewIdScheduler
from wbia_miew_id.helpers import get_config, write_config, update_bn
from wbia_miew_id.metrics import AverageMeter, compute_calibration
from wbia_miew_id.datasets import MiewIdDataset, get_train_transforms, get_valid_transforms
from wbia_miew_id.models import ArcMarginProduct, ElasticArcFace, ArcFaceSubCenterDynamic, MiewIdNet
from wbia_miew_id.engine import train_fn, eval_fn, group_eval_fn


from torch.optim.swa_utils import AveragedModel, SWALR
import torch
import random
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb


class Trainer:
    def __init__(self, config):
        self.config = config

    def set_seed_torch(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def run_fn(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None):
        best_score = 0
        for epoch in range(self.config.engine.epochs):
            train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=scheduler, epoch=epoch, use_wandb=use_wandb, swa_model=swa_model, swa_start=swa_start, swa_scheduler=swa_scheduler)

            print("\nGetting metrics on validation set...")

            eval_groups = self.config.data.test.eval_groups

            if eval_groups:
                valid_score, valid_cmc = group_eval_fn(self.config, eval_groups, model)
                print('Group average score: ', valid_score)
            else:
                print('Evaluating on full test set')
                valid_score = eval_fn(valid_loader, model, device, use_wandb=use_wandb)
                print('Valid score: ', valid_score)

            if valid_score > best_score:
                best_score = valid_score
                torch.save(model.state_dict(), f'{checkpoint_dir}/model_best.bin')
                print('best model found for epoch {}'.format(epoch))

        if swa_model:
            print("Updating SWA batchnorm statistics...")
            update_bn(train_loader, swa_model, device=device)
            torch.save(swa_model.state_dict(), f'{checkpoint_dir}/swa_model_{epoch}.bin')
            
        return best_score

    def run(self):
        config = self.config
        checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print('Checkpoints will be saved at: ', checkpoint_dir)

        config_path_out = f'{checkpoint_dir}/{config.exp_name}.yaml'
        config.data.test.checkpoint_path = f'{checkpoint_dir}/model_best.bin'

        self.set_seed_torch(config.engine.seed)

        df_train = preprocess_data(config.data.train.anno_path, 
                                    name_keys=config.data.name_keys,
                                    convert_names_to_ids=True, 
                                    viewpoint_list=config.data.viewpoint_list, 
                                    n_filter_min=config.data.train.n_filter_min, 
                                    n_subsample_max=config.data.train.n_subsample_max,
                                    use_full_image_path=config.data.use_full_image_path,
                                    images_dir=config.data.images_dir)

        df_val = preprocess_data(config.data.val.anno_path, 
                                  name_keys=config.data.name_keys,
                                  convert_names_to_ids=True, 
                                  viewpoint_list=config.data.viewpoint_list, 
                                  n_filter_min=config.data.val.n_filter_min, 
                                  n_subsample_max=config.data.val.n_subsample_max,
                                  use_full_image_path=config.data.use_full_image_path,
                                  images_dir=config.data.images_dir)


    
        print_intersect_stats(df_train, df_val, individual_key='name_orig')
    
        n_train_classes = df_train['name'].nunique()

        crop_bbox = config.data.crop_bbox

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

        train_dataset = MiewIdDataset(
            csv=df_train,
            transforms=get_train_transforms((config.data.image_size[0], config.data.image_size[1])),
            fliplr=config.test.fliplr,
            fliplr_view=config.test.fliplr_view,
            crop_bbox=crop_bbox)
        
        valid_dataset = MiewIdDataset(
            csv=df_val,
            transforms=get_valid_transforms((config.data.image_size[0], config.data.image_size[1])),
            fliplr=config.test.fliplr,
            fliplr_view=config.test.fliplr_view,
            crop_bbox=crop_bbox)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.engine.train_batch_size,
            num_workers=config.engine.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.engine.valid_batch_size,
            num_workers=config.engine.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        device = torch.device(config.engine.device)

        if config.model_params.n_classes != n_train_classes:
            print(f"WARNING: Overriding n_classes in config ({config.model_params.n_classes}) which is different from actual n_train_classes in the dataset - ({n_train_classes}).")
            config.model_params.n_classes = n_train_classes

        if config.model_params.loss_module == 'arcface_subcenter_dynamic':
            margin_min = 0.2
            margin_max = config.model_params.margin
            tmp = np.sqrt(1 / np.sqrt(df_train['name'].value_counts().sort_index().values))
            margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (margin_max - margin_min) + margin_min
        else:
            margins = None

        model = MiewIdNet(**dict(config.model_params), margins=margins)
        model.to(device)

        loss_fn = fetch_loss()

        if config.model_params.loss_module == 'elastic_arcface':
            criterion = ElasticArcFace(loss_fn=loss_fn, in_features=model.final_in_features, out_features=config.model_params.n_classes)
            criterion.to(device)

        elif config.model_params.loss_module == 'arcface_subcenter_dynamic':
            if margins is None:
                margins = [0.3] * n_classes
            criterion = ArcFaceSubCenterDynamic(
                    loss_fn=loss_fn,
                    embedding_dim=model.final_in_features, 
                    output_classes=config.model_params.n_classes, 
                    margins=margins,
                    s=config.model_params.s,
                    k=config.model_params.k)
            criterion.to(device)
        else:
            raise NotImplementedError("Loss module not recognized")

        optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=config.scheduler_params.lr_start)
        scheduler = MiewIdScheduler(optimizer, **dict(config.scheduler_params))

        if config.engine.use_swa:
            swa_model = AveragedModel(model)
            swa_model.to(device)
            swa_scheduler = SWALR(optimizer=optimizer, swa_lr=config.swa_params.swa_lr)
            swa_start = config.swa_params.swa_start
        else:
            swa_model = None
            swa_scheduler = None
            swa_start = None

        write_config(config, config_path_out)

        with WandbContext(config):
            best_score = self.run_fn(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=config.engine.use_wandb,
                                     swa_model=swa_model, swa_scheduler=swa_scheduler, swa_start=swa_start)

        return best_score

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file. Default: configs/default_config.yaml'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    config = get_config(config_path)
    trainer = Trainer(config)
    trainer.run()
