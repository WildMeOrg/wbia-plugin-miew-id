import cv2
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from .helpers import resize_image, unnormalize, get_chip_from_img

from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM, HiResCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)

def batch_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_cam_on_image(img: np.ndarray,
                        mask: np.ndarray,
                        use_rgb: bool = False,
                        colormap: int = cv2.COLORMAP_JET,
                        image_weight: float = 0.6) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)

    # Keep heatmap areas lower than threshold transparent
    heatmap[mask <= 0.05] = 0
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img

    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def draw_one(config, test_loader, model, images_dir = '', method='gradcam_plus_plus', eigen_smooth=False, show=False, use_cuda=True):

    # Generate embeddings for query and db
    model.eval()
    tk0 = tqdm(test_loader, total=len(test_loader))
    embeddings = []
    labels = []
    images = []
    paths = []
    bboxes = []
    with torch.no_grad():   
        for batch in tk0:
            batch_image = batch[0]
            batch_name = batch[1]
            batch_path = batch[2]
            batch_bbox = batch[3]


            images.extend(batch_image)
            batch_embeddings = model(batch_image.to(config.engine.device))
            
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            
            batch_embeddings_df = pd.DataFrame(batch_embeddings)
            embeddings.append(batch_embeddings_df)

            batch_labels = batch_name.tolist()
            labels.extend(batch_labels)
            
            paths.extend(batch_path)

            bboxes.extend(batch_bbox)

    bboxes = [t.int().tolist() for t in bboxes]
            
    embeddings = pd.concat(embeddings)

    target_layers = model.backbone.conv_head

    if method=='gradcam_plus_plus':
        generate_cam = GradCAMPlusPlus(model=model,target_layers=[target_layers],use_cuda=use_cuda)
    elif method=='eigencam':
        generate_cam = EigenCAM(model=model,target_layers=[target_layers],use_cuda=use_cuda)

    qry_idx = 0
    db_idx = 1

    qry_features = embeddings.iloc[qry_idx].values
    qry_features = torch.Tensor(qry_features).to(config.engine.device)

    db_features = embeddings.iloc[db_idx].values
    db_features = torch.Tensor(db_features).to(config.engine.device)

    similarity_to_qry = SimilarityToConceptTarget(qry_features)
    similarity_to_db = SimilarityToConceptTarget(db_features)

    qry_tensor = images[qry_idx]
    db_tensor = images[db_idx]

    db_tensor = db_tensor.unsqueeze(0)
    qry_tensor = qry_tensor.unsqueeze(0)


    # generate results
    stack_tensor = torch.cat([db_tensor, qry_tensor])
    stack_target = [similarity_to_qry, similarity_to_db]
    results_cam = generate_cam(input_tensor=stack_tensor,targets=stack_target,aug_smooth=False,eigen_smooth=eigen_smooth)
    qry_grayscale_cam = results_cam[0, :]
    db_grayscale_cam = results_cam[1, :]


    # query image results
    qry_image_path = paths[qry_idx]
    qry_float = load_image(qry_image_path)
    qry_bbox = bboxes[qry_idx]

    x1, y1, w, h = qry_bbox


    qry_float = qry_float[y1 : y1 + h, x1 : x1 + w]
    if min(qry_float.shape) < 1:
        # Use original image
        qry_float = qry_float = load_image(qry_image_path)

    qry_float_norm = (qry_float - qry_float.min()) / (qry_float.max() - qry_float.min())
    db_grayscale_cam_res = cv2.resize(db_grayscale_cam, (qry_float_norm.shape[1], qry_float_norm.shape[0]))
    cam_image_qry = show_cam_on_image(qry_float_norm, db_grayscale_cam_res, use_rgb=True)

    ai0 = cam_image_qry
    ai1 = qry_float

    # db image results
    db_image_path = paths[db_idx]
    db_float = load_image(db_image_path)
    db_bbox = bboxes[db_idx]
    x1, y1, w, h = db_bbox
    db_float = db_float[y1 : y1 + h, x1 : x1 + w]
    if min(db_float.shape) < 1:
        # Use original image
        db_float = db_float = load_image(db_image_path)

    db_float_norm = (db_float - db_float.min()) / (db_float.max() - db_float.min())
    qry_grayscale_cam_res = cv2.resize(qry_grayscale_cam, (db_float_norm.shape[1], db_float_norm.shape[0]))
    cam_image_db = show_cam_on_image(db_float_norm, qry_grayscale_cam_res, use_rgb=True)

    ai2 = cam_image_db
    ai3 = db_float

    image_list = [ai0, ai1, ai2, ai3]
    resize_height = 440
    resized_image_list = [resize_image(img, resize_height) for img in image_list]
    comb_image = np.hstack(resized_image_list)
    if show:
        plt.imshow(comb_image)

    comb_image = cv2.cvtColor(comb_image, cv2.COLOR_BGR2RGB)
    return comb_image

def generate_embeddings(device, model, test_loader):
    print('generating embeddings')
    tk0 = tqdm(test_loader, total=len(test_loader))
    embeddings = []
    labels = []
    images = []
    paths = []
    bboxes = []
    thetas = []
    with torch.no_grad():   
        for batch in tk0:

            try:
                batch_image = batch[0]
                batch_name = batch[1]
                batch_path = batch[2]
                batch_bbox = batch[3]
                batch_theta = batch[4]
            except:
                batch_image = batch['image']
                batch_name = batch['label']
                batch_path = batch["file_path"]
                batch_bbox = batch["bbox"]
                batch_theta = batch["theta"]


            images.extend(batch_image)
            batch_embeddings = model(batch_image.to(device))
            
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            
            batch_embeddings_df = pd.DataFrame(batch_embeddings)
            embeddings.append(batch_embeddings_df)

            batch_labels = batch_name.tolist()
            labels.extend(batch_labels)
            
            paths.extend(batch_path)
            bboxes.extend(batch_bbox)
        
            batch_theta = batch_theta.tolist()
            thetas.extend(batch_theta)


    bboxes = [t.int().tolist() for t in bboxes]

    embeddings = pd.concat(embeddings)
    return embeddings, labels, images, paths, bboxes, thetas

def draw_batch(device, test_loader, model, images_dir = '', method='hires_cam', eigen_smooth=False, render_transformed=False, show=False, use_cuda=True):

    print('** draw_batch started')

    # Generate embeddings for query and db
    model.eval()

    embeddings, labels, images, paths, bboxes, thetas = generate_embeddings(device, model, test_loader)

    print('*** embeddings generated ***')

    target_layers = model.backbone.conv_head

    if method=='gradcam_plus_plus':
        generate_cam = GradCAMPlusPlus(model=model,target_layers=[target_layers],use_cuda=use_cuda)
    elif method=='eigencam':
        generate_cam = EigenCAM(model=model,target_layers=[target_layers],use_cuda=use_cuda)
    elif method=='hires_cam':
        generate_cam = HiResCAM(model=model,target_layers=[target_layers],use_cuda=use_cuda)
    elif method=='gradcam':
        generate_cam = GradCAM(model=model,target_layers=[target_layers],use_cuda=use_cuda)

    qry_idx = 0
    db_idx = 1

    qry_features = embeddings.iloc[qry_idx].values
    qry_features = torch.Tensor(qry_features).to(device)

    db_features_batch = embeddings.iloc[db_idx:].values
    db_features_batch = torch.Tensor(db_features_batch).to(device)

    tensors = []
    stack_target = []    
    print('generating similarity targets')
    for i, db_features in tqdm(enumerate(db_features_batch)):

        similarity_to_qry = SimilarityToConceptTarget(qry_features)
        similarity_to_db = SimilarityToConceptTarget(db_features)

        qry_tensor = images[qry_idx]
        db_tensor = images[db_idx + i]

        db_tensor = db_tensor.unsqueeze(0)
        qry_tensor = qry_tensor.unsqueeze(0)
        tensors.extend([db_tensor, qry_tensor])
        stack_target.extend([similarity_to_qry, similarity_to_db])

    stack_tensor = torch.cat(tensors)

    batch_images = []
    results_cam = []
    
    batch_size = test_loader.batch_size

    batch_step = 2
    print('generating cams')

    for i in tqdm(range(0, len(stack_target), batch_step)):
        stack_tensor_batch = stack_tensor[i:i+batch_step]
        stack_target_batch = stack_target[i:i+batch_step]
        results_cam_batch = generate_cam(input_tensor=stack_tensor_batch,targets=stack_target_batch,aug_smooth=False,eigen_smooth=eigen_smooth)
        results_cam.extend(results_cam_batch)

    results_cam = np.array(results_cam)

    print('generating image tile')
    for i in tqdm(range(0, results_cam.shape[0], 2)):
        qry_grayscale_cam = results_cam[i, :]
        db_grayscale_cam = results_cam[i+1, :]

        # query image results
        if render_transformed:
            qry_tensor_unnorm = unnormalize(qry_tensor)
            qry_float = qry_tensor_unnorm.numpy().squeeze().transpose(1,2,0)
        else:
            qry_image_path = paths[qry_idx]
            qry_float = load_image(qry_image_path)
            qry_bbox = bboxes[qry_idx]
            # x1, y1, w, h = qry_bbox
            # qry_float = qry_float[y1 : y1 + h, x1 : x1 + w]
            # if min(qry_float.shape) < 1:
            #     # Use original image
            #     qry_float = qry_float = load_image(qry_image_path)
            qry_theta = thetas[qry_idx]
            qry_float = get_chip_from_img(qry_float, qry_bbox, qry_theta)


        qry_float_norm = (qry_float - qry_float.min()) / (qry_float.max() - qry_float.min())
        db_grayscale_cam_res = cv2.resize(db_grayscale_cam, (qry_float_norm.shape[1], qry_float_norm.shape[0]))
        cam_image_qry = show_cam_on_image(qry_float_norm, db_grayscale_cam_res, use_rgb=True)

        ai0 = cam_image_qry
        ai1 = qry_float

        # db image results
        if render_transformed:
            db_tensor_unnorm = unnormalize(stack_tensor[i])
            db_float = db_tensor_unnorm.numpy().squeeze().transpose(1,2,0)
        else:
            db_image_path = paths[db_idx + i//2]
            db_float = load_image(db_image_path)
            db_bbox = bboxes[db_idx + i//2]
            # x1, y1, w, h = db_bbox
            # db_float = db_float[y1 : y1 + h, x1 : x1 + w]
            # if min(db_float.shape) < 1:
            #     # Use original image
            #     db_float = db_float = load_image(db_image_path)
            db_theta = thetas[db_idx + i//2]
            db_float = get_chip_from_img(db_float, db_bbox, db_theta)


        db_float_norm = (db_float - db_float.min()) / (db_float.max() - db_float.min())
        qry_grayscale_cam_res = cv2.resize(qry_grayscale_cam, (db_float_norm.shape[1], db_float_norm.shape[0]))
        cam_image_db = show_cam_on_image(db_float_norm, qry_grayscale_cam_res, use_rgb=True)

        ai2 = cam_image_db
        ai3 = db_float

        image_list = [ai0, ai1, ai2, ai3]
        resize_height = 440
        resized_image_list = [resize_image(img, resize_height) for img in image_list]

        if render_transformed:
            ai0_float = ai0.astype(np.float32) / 255
            ai2_float = ai2.astype(np.float32) / 255
            image_list = [ai0_float, ai1, ai2_float, ai3]
            resized_image_list = image_list.copy()

        comb_image = np.hstack(resized_image_list)
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(comb_image)
            plt.show()

        comb_image = cv2.cvtColor(comb_image, cv2.COLOR_BGR2RGB)

        batch_images.append(comb_image)
    return batch_images