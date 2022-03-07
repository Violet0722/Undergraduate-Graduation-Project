import torch
import torch.nn.functional as F
import numpy as np


def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)


def torch_to_numpy(a: torch.Tensor):
    return a.squeeze(0).permute(1,2,0).numpy()


def sample_patch_transformed(im, pos, scale, image_sz, transforms, is_mask=False):
    """Extract transformed image samples.
    args:
        im: Image.
        pos: Center position for extraction.
        scale: Image scale to extract features from.
        image_sz: Size to resize the image samples to before extraction.
        transforms: A set of image transforms to apply.
    """

    # Get image patche
    im_patch, _ = sample_patch(im, pos, scale*image_sz, image_sz, is_mask=is_mask)

    # Apply transforms
    im_patches = torch.cat([T(im_patch, is_mask=is_mask) for T in transforms])

    return im_patches


def sample_patch_multiscale(im, pos, scales, image_sz, mode: str='replicate', max_scale_change=None):
    """Extract image patches at multiple scales.
    args:
        im: Image.
        pos: Center position for extraction.
        scales: Image scales to extract image patches from.
        image_sz: Size to resize the image samples to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """
    if isinstance(scales, (int, float)):
        scales = [scales]

    # Get image patches
    patch_iter, coord_iter = zip(*(sample_patch(im, pos, s*image_sz, image_sz, mode=mode,
                                                max_scale_change=max_scale_change) for s in scales))
    im_patches = torch.cat(list(patch_iter))
    patch_coords = torch.cat(list(coord_iter))

    return im_patches, patch_coords


def sample_patch(im: torch.Tensor, pos: torch.Tensor, sample_sz: torch.Tensor, output_sz: torch.Tensor = None,
                 mode: str = 'replicate', max_scale_change=None, is_mask=False):
    """Sample an image patch.
       通过尺度ｓｃａｌｅｓ得到搜索区域
    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
        mode: how to treat image borders: 'replicate' (default), 'inside' or 'inside_major'
        max_scale_change: maximum allowed scale change when using 'inside' and 'inside_major' mode
    """

    # if mode not in ['replicate', 'inside']:
    #     raise ValueError('Unknown border mode \'{}\'.'.format(mode))

    # copy and convert
    posl = pos.long().clone()

    pad_mode = mode

    #print("preprocessing.py------im----{}--pos-----{}--saple_sz-----{}--output_sz-----{}".format(im.shape,pos,sample_sz,output_sz))

    # Get new sample size if forced inside the image
    # 如果强制放入图像，则获取新的样本大小
    if mode == 'inside' or mode == 'inside_major':

        pad_mode = 'replicate'
        im_sz = torch.Tensor([im.shape[2], im.shape[3]])
        shrink_factor = (sample_sz.float() / im_sz)  # 收缩系数
        if mode == 'inside':
            shrink_factor = shrink_factor.max()
        elif mode == 'inside_major':
            shrink_factor = shrink_factor.min()
        # clamp判断上下限，小于min则取min值，大于max_scale_change则取max_scale_change，在两个中间还是原来值
        shrink_factor.clamp_(min=1, max=max_scale_change)
        sample_sz = (sample_sz.float() / shrink_factor).long()



    # Compute pre-downsampling factor计算预下采样系数
    if output_sz is not None:
        resize_factor = torch.min(sample_sz.float() / output_sz.float()).item()
        df = int(max(int(resize_factor - 0.1), 1))
        #print("*******",resize_factor,df)
    else:
        df = int(1)

    sz = sample_sz.float() / df     # new size
    #print("newsize*****",sz)
    # Do downsampling
    if df > 1:
        os = posl % df              # offset
        posl = (posl - os) / df     # new position
        im2 = im[..., os[0].item()::df, os[1].item()::df]   # downsample
    else:
        im2 = im

    #print("******",im2.shape)
    # compute size to crop
    szl = torch.max(sz.round(), torch.Tensor([2])).long()
    #print("*******",szl)
    # Extract top and bottom coordinates　　提取左上角和右下角坐标
    tl = posl - (szl - 1)/2
    br = posl + szl/2 + 1

    # Shift the crop to inside
    if mode == 'inside' or mode == 'inside_major':
        im2_sz = torch.LongTensor([im2.shape[2], im2.shape[3]])
        shift = (-tl).clamp(0) - (br - im2_sz).clamp(0)
        tl += shift
        br += shift

        outside = ((-tl).clamp(0) + (br - im2_sz).clamp(0)) // 2
        shift = (-tl - outside) * (outside > 0).long()
        tl += shift
        br += shift

        # Get image patch
        # im_patch = im2[...,tl[0].item():br[0].item(),tl[1].item():br[1].item()]

    # Get image patch
    if not is_mask:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]), pad_mode)
    else:
        im_patch = F.pad(im2, (-tl[1].item(), br[1].item() - im2.shape[3], -tl[0].item(), br[0].item() - im2.shape[2]))

    # Get image coordinates得到图像坐标左上和右下角的坐标
    patch_coord = df * torch.cat((tl, br)).view(1,4)

    if output_sz is None or (im_patch.shape[-2] == output_sz[0] and im_patch.shape[-1] == output_sz[1]):
        return im_patch.clone(), patch_coord

    # Resample
    if not is_mask:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='bilinear')
    else:
        im_patch = F.interpolate(im_patch, output_sz.long().tolist(), mode='nearest')

    return im_patch, patch_coord
