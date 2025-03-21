from tqdm import tqdm
from pairx.core import explain
from pairx.xai_dataset import get_pretransform_img


def draw_one(
    config,
    test_loader,
    model,
    images_dir="",
    method="gradcam_plus_plus",
    eigen_smooth=False,
    show=False,
    use_cuda=True,
    visualization_type="lines_and_colors",
    layer_key="backbone.blocks.3",
    k_lines=20,
    k_colors=10,
):
    """
    Generates a PAIR-X explanation for the provided images and model.

    Args:
        config: Config object containing device (accessed as config.engine.device).
        test_loader (DataLoader): Should contain two images, with 4 items for each (image, name, path, bbox as xywh).
        model (torch.nn.Module or equivalent): The deep metric learning model.
        images_dir: ignored
        method: ignored
        eigen_smooth: ignored
        show: ignored
        use_cuda: ignored
        visualization_type (str): The part of the PAIR-X visualization to return, selected from "lines_and_colors" (default), "only_lines", and "only_colors".
        layer_keys (str): The key of the intermediate layer to be used for explanation. Defaults to 'backbone.blocks.3'.
        k_lines (int, optional): The number of matches to visualize as lines. Defaults to 20.
        k_colors (int, optional): The number of matches to backpropagate to original image pixels. Defaults to 10.

    Returns:
        numpy.ndarray: PAIR-X visualization of type visualization_type.
    """
    assert len(test_loader) == 2, "test_loader should only contain two images"
    assert visualization_type in (
        "lines_and_colors",
        "only_lines",
        "only_colors",
    ), "unsupported visualization type"

    # get transformed and untransformed images out of test_loader
    transformed_images = []
    pretransform_images = []

    for batch in test_loader:
        transformed_image, _, path, bbox = batch

        transformed_images.append(transformed_image.to(config.engine.device))

        img_size = tuple(transformed_image.shape[-2:])
        pretransform_image = get_pretransform_img(path, img_size, bbox)
        pretransform_images.append(pretransform_image)

    img_0, img_1 = transformed_images
    img_np_0, img_np_1 = pretransform_images

    # If only returning image with lines, skip generating color maps to save time
    if visualization_type == "only_lines":
        k_colors = 0

    # generate explanation image and return
    model.eval()
    pairx_img = explain(
        img_0,
        img_1,
        img_np_0,
        img_np_1,
        model,
        [layer_key],
        k_lines=k_lines,
        k_colors=k_colors,
    )

    pairx_height = pairx_img.shape[0] // 2

    if visualization_type == "only_lines":
        return pairx_img[:pairx_height]
    elif visualization_type == "only_colors":
        return pairx_img[pairx_height:]

    return pairx_img
