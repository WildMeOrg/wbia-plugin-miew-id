from wbia_miew_id.datasets import MiewIdDataset, get_train_transforms, get_valid_transforms, get_test_transforms
from wbia_miew_id.logging_utils import WandbContext
from wbia_miew_id.models import MiewIdNet
from wbia_miew_id.etl import preprocess_data, print_basic_stats
from wbia_miew_id.engine import eval_fn, group_eval_run
from wbia_miew_id.helpers import get_config
from wbia_miew_id.visualization import render_query_results
from wbia_miew_id.metrics import precision_at_k

import os
import torch
import random
import numpy as np

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file. Default: configs/default_config.yaml'
    )

    parser.add_argument('--visualize', '--vis', action='store_true')

    return parser.parse_args()

class Evaluator:
    def __init__(self, device, seed, anno_path, name_keys, viewpoint_list, use_full_image_path, images_dir, image_size,
                 crop_bbox, valid_batch_size, num_workers, eval_groups, fliplr, fliplr_view, n_filter_min, n_subsample_max,
                 model_params=None, checkpoint_path=None, model=None, visualize=False, visualization_output_dir='miewid_visualizations'):
        self.device = device
        self.visualize = visualize
        self.seed = seed
        self.model_params = model_params
        self.checkpoint_path = checkpoint_path
        self.anno_path = anno_path
        self.name_keys = name_keys
        self.viewpoint_list = viewpoint_list
        self.use_full_image_path = use_full_image_path
        self.images_dir = images_dir
        self.image_size = image_size
        self.crop_bbox = crop_bbox
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        self.eval_groups = eval_groups
        self.fliplr = fliplr
        self.fliplr_view = fliplr_view
        self.n_filter_min = n_filter_min
        self.n_subsample_max = n_subsample_max
        self.visualization_output_dir = visualization_output_dir

        self.set_seed_torch(seed)
        
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = self.load_model(device, model_params, checkpoint_path)
    
    @staticmethod
    def set_seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def load_model(device, model_params, checkpoint_path):
        model = MiewIdNet(**model_params)
        model.to(device)
        
        if checkpoint_path:
            weights = torch.load(checkpoint_path, map_location=device)
            n_train_classes = weights[list(weights.keys())[-1]].shape[-1]
            if model_params['n_classes'] != n_train_classes:
                print(f"WARNING: Overriding n_classes in config ({model_params['n_classes']}) which is different from actual n_train_classes in the checkpoint -  ({n_train_classes}).")
                model_params['n_classes'] = n_train_classes
            model.load_state_dict(weights, strict=False)
            print('loaded checkpoint from', checkpoint_path)
        
        return model

    @staticmethod
    def preprocess_test_data(anno_path, name_keys, viewpoint_list, use_full_image_path, 
                             images_dir, image_size, crop_bbox, valid_batch_size, num_workers, 
                             fliplr, fliplr_view, n_filter_min, n_subsample_max):
        df_test = preprocess_data(
            anno_path, 
            name_keys=name_keys,
            convert_names_to_ids=True, 
            viewpoint_list=viewpoint_list, 
            n_filter_min=n_filter_min, 
            n_subsample_max=n_subsample_max,
            use_full_image_path=use_full_image_path,
            images_dir=images_dir,
        )
        
        test_dataset = MiewIdDataset(
            csv=df_test,
            transforms=get_test_transforms((image_size[0], image_size[1])),
            fliplr=fliplr,
            fliplr_view=fliplr_view,
            crop_bbox=crop_bbox,
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=valid_batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        return test_loader, df_test

    @staticmethod
    def evaluate_groups(self, eval_groups, anno_path, name_keys, viewpoint_list, 
                        use_full_image_path, images_dir, model):
        df_test_group = preprocess_data(
            anno_path,
            name_keys=name_keys,
            convert_names_to_ids=True, 
            viewpoint_list=viewpoint_list, 
            n_filter_min=None, 
            n_subsample_max=None,
            use_full_image_path=use_full_image_path,
            images_dir=images_dir
        )
        group_results = group_eval_run(df_test_group, eval_groups, model,
        n_filter_min = self.n_filter_min, 
        n_subsample_max = self.n_subsample_max, 
        image_size = self.image_size, 
        fliplr = self.fliplr, 
        fliplr_view = self.fliplr_view, 
        crop_bbox = self.crop_bbox, 
        valid_batch_size = self.valid_batch_size, 
        device = self.device)

    @staticmethod
    def visualize_results(test_outputs, df_test, test_dataset, model, device, k=5, valid_batch_size=2, output_dir='miewid_visualizations'):
        embeddings, q_pids, distmat = test_outputs
        ranks = list(range(1, k+1))
        score, match_mat, topk_idx, topk_names = precision_at_k(q_pids, distmat, ranks=ranks, return_matches=True)
        match_results = (q_pids, topk_idx, topk_names, match_mat)
        render_query_results(model, test_dataset, df_test, match_results, device,
                             k=k, valid_batch_size=valid_batch_size, output_dir=output_dir)

    def evaluate(self):
        test_loader, df_test = self.preprocess_test_data(
            self.anno_path, self.name_keys, self.viewpoint_list, 
            self.use_full_image_path, self.images_dir, self.image_size, self.crop_bbox, 
            self.valid_batch_size, self.num_workers, self.fliplr, 
            self.fliplr_view, self.n_filter_min, self.n_subsample_max
        )
        test_score, cmc, test_outputs = eval_fn(test_loader, self.model, self.device, use_wandb=False, return_outputs=True)

        if self.eval_groups:
            self.evaluate_groups(self,
                self.eval_groups, self.anno_path, self.name_keys, 
                self.viewpoint_list, self.use_full_image_path, 
                self.images_dir, self.model
            )

        if self.visualize:
            self.visualize_results(test_outputs, df_test, test_loader.dataset, self.model, self.device,
                                  k=5, valid_batch_size=self.valid_batch_size,output_dir=self.visualization_output_dir )

        return test_score

if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config)

    visualization_output_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}/visualizations"
    
    evaluator = Evaluator(
    device=torch.device(config.engine.device),
    seed=config.engine.seed,
    anno_path=config.data.test.anno_path,
    name_keys=config.data.name_keys,
    viewpoint_list=config.data.viewpoint_list,
    use_full_image_path=config.data.use_full_image_path,
    images_dir=config.data.images_dir,
    image_size=(config.data.image_size[0], config.data.image_size[1]),
    crop_bbox=config.data.crop_bbox,
    valid_batch_size=config.engine.valid_batch_size,
    num_workers=config.engine.num_workers,
    eval_groups=config.data.test.eval_groups,
    fliplr=config.test.fliplr,
    fliplr_view=config.test.fliplr_view,
    n_filter_min=config.data.test.n_filter_min,
    n_subsample_max=config.data.test.n_subsample_max,
    model_params=dict(config.model_params),
    checkpoint_path=config.data.test.checkpoint_path,
    model=None,
    visualize=args.visualize,
    visualization_output_dir=visualization_output_dir
)
    
    evaluator.evaluate()