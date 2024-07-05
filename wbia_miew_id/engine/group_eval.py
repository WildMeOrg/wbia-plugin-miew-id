import torch
import numpy as np
from wbia_miew_id.datasets import MiewIdDataset, get_test_transforms
from wbia_miew_id.engine import eval_fn, log_results
from wbia_miew_id.etl import filter_min_names_df, subsample_max_df, preprocess_data


def group_eval_run(df_test, eval_groups, model, n_filter_min, n_subsample_max, image_size, fliplr, fliplr_view, crop_bbox, valid_batch_size, device):

    print("** Calculating groupwise evaluation scores **")

    group_results = []
    for eval_group in eval_groups:
        for group, df_group in df_test.groupby(eval_group):
            try:
                print('* Evaluating group:', group)
                if n_filter_min:
                    print(len(df_group))
                    df_group = filter_min_names_df(df_group, n_filter_min)
                if n_subsample_max:
                    df_group = subsample_max_df(df_group, n_subsample_max)
                test_dataset = MiewIdDataset(
                    csv=df_group,
                    transforms=get_test_transforms((image_size[0], image_size[1])),
                    fliplr=fliplr,
                    fliplr_view=fliplr_view,
                    crop_bbox=crop_bbox,
                )
                    
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=valid_batch_size,
                    num_workers=0,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                )
                device = torch.device(device)
                test_score, test_cmc, test_outputs = eval_fn(test_loader, model, device, use_wandb=False, return_outputs=True)
            except Exception as E:
                print('* Could not evaluate group:', group)
                print(E)
                test_score, test_cmc = 0, [0]*20

            group_result = (group, test_score, test_cmc)

            group_results.append(group_result)

    return group_results


def group_eval_fn(config, eval_groups, model, use_wandb=True):
    print('Evaluating on groups')
    df_test_group = preprocess_data(config.data.test.anno_path, 
                        name_keys=config.data.name_keys,
                        convert_names_to_ids=True, 
                        viewpoint_list=config.data.viewpoint_list, 
                        n_filter_min=None, 
                        n_subsample_max=None,
                        use_full_image_path=config.data.use_full_image_path,
                        images_dir = config.data.images_dir)
    group_results = group_eval_run(df_test_group, eval_groups, model,
        n_filter_min = config.data.test.n_filter_min, 
        n_subsample_max = config.data.test.n_subsample_max, 
        image_size = (config.data.image_size[0], config.data.image_size[1]), 
        fliplr = config.test.fliplr, 
        fliplr_view = config.test.fliplr_view, 
        crop_bbox = config.data.crop_bbox, 
        valid_batch_size = config.engine.valid_batch_size, 
        device = config.engine.device)

    group_scores = []
    group_cmcs = []

    for (group, group_score, group_cmc) in group_results:
        group_tag = '-'.join(group) if isinstance(group, tuple) else group
        log_results(group_score, group_cmc, group_tag, use_wandb=use_wandb)

        group_scores.append(group_score)
        group_cmcs.append(group_cmc)

    group_scores = [x for x in group_scores if x != 0]
    valid_score = np.mean(group_scores)
    valid_cmc = np.mean(group_cmcs, axis=0).tolist()

    log_results(valid_score, valid_cmc, 'Avg', use_wandb=use_wandb)

    return valid_score, valid_cmc