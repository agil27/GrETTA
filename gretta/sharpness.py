import torch
import torch.nn.functional as F


def _to_bchw(tensor, color_channel_num=None):
    """Converts a PyTorch tensor image to BCHW format.

    Args:
        tensor (torch.Tensor): image of the form :math:`(H, W)`, :math:`(C, H, W)`, :math:`(H, W, C)` or
            :math:`(B, C, H, W)`.
        color_channel_num (Optional[int]): Color channel of the input tensor.
            If None, it will not alter the input channel.

    Returns:
        torch.Tensor: input tensor of the form :math:`(B, H, W, C)`.
    """
    if not torch.is_tensor(tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError(f"Input size must be a two, three or four dimensional tensor. Got {tensor.shape}")

    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    if color_channel_num is not None and color_channel_num != 1:
        channel_list = [0, 1, 2, 3]
        channel_list.insert(1, channel_list.pop(color_channel_num))
        tensor = tensor.permute(*channel_list)
    return tensor



def sharpness(input, factor):
    r"""Implements Sharpness function from PIL using torch ops.

    Args:
        input (torch.Tensor): image tensor with shapes like (C, H, W) or (B, C, H, W) to sharpen.
        factor (float or torch.Tensor): factor of sharpness strength. Must be above 0.
            If float or one element tensor, input will be sharpened by the same factor across the whole batch.
            If 1-d tensor, input will be sharpened element-wisely, len(factor) == len(input).

    Returns:
        torch.Tensor: Sharpened image or images.
    """
    input = _to_bchw(input)
    if isinstance(factor, torch.Tensor):
        factor = factor.squeeze()
        if len(factor.size()) != 0:
            assert input.size(0) == factor.size(0), \
                f"Input batch size shall match with factor size if 1d array. Got {input.size(0)} and {factor.size(0)}"
    else:
        factor = float(factor)
    kernel = torch.tensor([
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ], dtype=input.dtype, device=input.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    # This shall be equivalent to depthwise conv2d:
    # Ref: https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315/2
    degenerate = F.conv2d(input, kernel, bias=None, stride=1, groups=input.size(1))
    degenerate = torch.clamp(degenerate, 0., 1.)

    mask = torch.ones_like(degenerate)
    padded_mask = F.pad(mask, [1, 1, 1, 1])
    padded_degenerate = F.pad(degenerate, [1, 1, 1, 1])
    result = torch.where(padded_mask == 1, padded_degenerate, input)

    def _blend_one(input1, input2, factor):
        if isinstance(factor, torch.Tensor):
            factor = factor.squeeze()
            assert len(factor.size()) == 0, f"Factor shall be a float or single element tensor. Got {factor}"
        if factor == 0.:
            return input1
        if factor == 1.:
            return input2
        diff = (input2 - input1) * factor
        res = input1 + diff
        if factor > 0. and factor < 1.:
            return res
        return torch.clamp(res, 0, 1)
    if isinstance(factor, (float)) or len(factor.size()) == 0:
        return _blend_one(input, result, factor)
    return torch.stack([_blend_one(input[i], result[i], factor[i]) for i in range(len(factor))])