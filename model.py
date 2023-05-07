import torch
import torch.nn as nn


def model_transference(
        pre_trained: nn.Module,
        output_shape: int,
        load_model: str = None,
        eval: bool = False,
        device: str = 'cpu',
):
    model = pre_trained

    # Freeze the weights
    for param in model.parameters():
        param.requires_grad = False

    # Get fully-connected layer input shape
    num_ftrs = model.fc.in_features

    # Replace with new output shape
    model.fc = nn.Linear(in_features=num_ftrs,
                         out_features=output_shape)

    # Load model state dict
    model.load_state_dict(torch.load(load_model)) if load_model else None

    # Evaluate mode if eval = True    
    model.eval() if eval else None
    
    # To device
    model.to(device)

    return model
