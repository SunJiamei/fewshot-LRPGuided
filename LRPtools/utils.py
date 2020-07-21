import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage.transform
LOWEST = -1
HIGHEST = 1
EPSILON = 0.01
Z_EPSILON = 1e-7
LOGIT_BETA = 4

def safe_divide(numerator, divisor):
    return numerator / (divisor + Z_EPSILON * (divisor == 0).float())


def project(X, output_range=(0, 1), absmax=None, input_is_postive_only=False):

    if absmax is None:
        absmax = np.max(np.abs(X),
                        axis=tuple(range(1, len(X.shape))))
    absmax = np.asarray(absmax)
    # print(absmax.shape)
    # print(X.shape)
    mask = absmax != 0
    # print(mask)
    if mask.sum() > 0:
        X[mask] /= absmax[mask]

    if input_is_postive_only is False:
        X = (X+1)/2  # [0, 1]
    X = X.clip(0, 1)

    X = output_range[0] + (X * (output_range[1]-output_range[0]))
    return X


def normalize_relevance(X, dim=-1):

    value, indice = torch.max(torch.abs(X), dim=dim)
    value.masked_fill_(value==0, 1)
    X = X/value.unsqueeze(dim)
    return X + 1





def heatmap(X, cmap_type="seismic", reduce_op="sum", reduce_axis=-1, **kwargs):
    cmap = plt.cm.get_cmap(cmap_type)

    tmp = X
    shape = tmp.shape

    if reduce_op == "sum":
        tmp = tmp.sum(axis=reduce_axis)
    elif reduce_op == "absmax":
        pos_max = tmp.max(axis=reduce_axis)
        neg_max = (-tmp).max(axis=reduce_axis)
        abs_neg_max = -neg_max
        tmp = np.select([pos_max >= abs_neg_max, pos_max < abs_neg_max],
                        [pos_max, neg_max])
    else:
        raise NotImplementedError()
    tmp = project(tmp, output_range=(0, 255), **kwargs).astype(np.int64)

    tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[reduce_axis] = 3
    return tmp.reshape(shape).astype(np.float32)


def graymap(X, **kwargs):
    return heatmap(X, cmap_type="gray", **kwargs)


def gamma(X, gamma = 0.5, minamp=0, maxamp=None):
    """
    apply gamma correction to an input array X
    while maintaining the relative order of entries,
    also for negative vs positive values in X.
    the fxn firstly determines the max
    amplitude in both positive and negative
    direction and then applies gamma scaling
    to the positive and negative values of the
    array separately, according to the common amplitude.

    :param gamma: the gamma parameter for gamma scaling
    :param minamp: the smallest absolute value to consider.
    if not given assumed to be zero (neutral value for relevance,
        min value for saliency, ...). values above and below
        minamp are treated separately.
    :param maxamp: the largest absolute value to consider relative
    to the neutral value minamp
    if not given determined from the given data.
    """

    Y = np.zeros_like(X)

    X = X - minamp # shift to given/assumed center
    if maxamp is None: maxamp = np.abs(X).max() #infer maxamp if not given
    X = X / maxamp # scale linearly

    #apply gamma correction for both positive and negative values.
    i_pos = X >= 0
    i_neg = np.invert(i_pos)
    Y[i_pos] = X[i_pos]**gamma
    Y[i_neg] = -(-X[i_neg])**gamma

    #reconstruct original scale and center
    Y *= maxamp
    Y += minamp

    return Y


def visuallize_attention(image, attention, reshape_size, cmap_type="seismic",):
    '''this function will blend the original highlightened by the attention
        the input images are with shape (channel, height, width) an Image object
        the attentions are with shape (1, height*width)) a pytorch tensor'''

    def project_inside(x):
        absmax = np.max(np.abs(x))
        x = 1.0 * x / absmax
        if np.sum(x < 0):
            x = (x + 1) / 2
        else:
            x = x
        return x
    attention = attention.view(reshape_size)
    attention = attention.cpu().detach().numpy()-1
    attention = project_inside(attention)
    atn = skimage.transform.pyramid_expand(attention, upscale=14,
                                           multichannel=False)
    cm = plt.get_cmap('jet')
    atn_heatmap = cm(atn)
    attention_heatmap = Image.fromarray(np.uint8(atn_heatmap[:, :, :3]*255))
    merged_heatmap = Image.blend(image, attention_heatmap, 0.5)
    return merged_heatmap




def compute_lrp_sum(sum_output, sum_input, relevance_sum_output,dim=-1):
    #this function will return the relevance of the sum_input
    assert (sum_output == torch.sum(sum_input,dim=dim)).all()
    fea_dim = sum_input.size()[-1]
    relevance = relevance_sum_output.unsqueeze(-1).repeat(1,1,1,fea_dim)
    out = sum_output.unsqueeze(-1).repeat(1,1,1,fea_dim)
    mask = out == 0
    out.masked_fill_(mask, 1 / fea_dim)
    relevance_sum_input = relevance * sum_input / (out + util.EPSILON * out.sign())
    return relevance_sum_input

def compute_lrp_mean(mean_output, mean_input, relevance_mean_output,dim=-1):
    #this function will return the relevance of the mean_input
    assert (mean_output == torch.mean(mean_input,dim=dim)).all()
    fea_dim = mean_input.size()[-1]
    input_dim = len(mean_input.shape)
    repeat_param = [1]*input_dim
    repeat_param[-1] *= fea_dim
    relevance = relevance_mean_output.unsqueeze(-1).repeat(repeat_param)
    out = mean_input.sum(dim=dim).unsqueeze(-1).repeat(repeat_param)
    mask = out == 0
    out.masked_fill_(mask, 1/fea_dim)
    relevance_mean_input = relevance * mean_input / (out + util.EPSILON * out.sign())
    return relevance_mean_input
