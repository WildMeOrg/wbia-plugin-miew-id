import torch

# We need to define a custom update_bn because the dataloaders
# iterate through dicts rather than 2-tuples of (image, label) pairs
@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, dict):
            label = input['label']
            input = input['image']
        if device is not None:
            input.to(device)
            label.to(device)

        if was_training:
            model(input, label)
        else:
            model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

def extract_outputs(net, data_loader, checkpoint=None, device="cpu"):
    """
    Extract outputs and targets of the given model on the entire dataloader, 
    optionally loading weights from a checkpoint.
    Returns:
        model outputs, data_loader targets
    """
    # Load model checkpoint if provided
    if checkpoint:
        saved_state_dict = torch.load(checkpoint)
        # Handle both new and legacy checkpoint formats
        if isinstance(saved_state_dict, dict) and 'model_state_dict' in saved_state_dict:
            net.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            net.load_state_dict(saved_state_dict)
    
    net = net.to(device)
    net.eval()

    model_outputs, all_targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['image']
            targets = batch['label']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net.extract_logits(inputs, targets)

            model_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    model_outputs = torch.cat(model_outputs, dim=0)
    all_targets = torch.cat(all_targets)
    return model_outputs, all_targets