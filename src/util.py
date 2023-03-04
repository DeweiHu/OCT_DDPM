# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:16:56 2020

@author: hudew
"""

import nibabel as nib
import numpy as np
import os
from PIL import Image

def nii_loader(dir):
    data_nii = nib.load(dir)
    data = np.array(data_nii.dataobj)
    return data

def nii_saver(volume,path,filename,header=None):
    output = nib.Nifti1Image(volume, np.eye(4), header=header)
    nib.save(output,os.path.join(path,filename))

def nii_merge(file_a,file_b,axis):
    Va = nii_loader(file_a)
    Vb = nii_loader(file_b)
    V = np.concatenate([Va,Vb],axis=axis)
    return V

def ImageRescale(im, I_range):
    im_range = im.max() - im.min()
    target_range = I_range[1] - I_range[0]
    
    if im_range == 0:
        target = np.zeros(im.shape, dtype=np.float32)
    else:
        target = I_range[0] + target_range/im_range * (im - im.min())
    return np.float32(target)


def ImageCrop(img,p1,p2):
    '''
    Take a triangular region out by given 2 diagonal points
    '''
    opt = img[p1[0]:p2[0],p1[1]:p2[1]]
    return opt

def rot(img,dir):
    if dir == 'cw':
        opt = np.fliplr(np.transpose(img))
    elif dir == 'ccw':
        opt = np.flipud(np.transpose(img))
    else:
        print('Direction undefined')
    return opt

'''
import shape: [slc,H,W]
'''
def itkshape(v):
    v = np.transpose(v,(0,2,1))
    h,w,slc = v.shape
    opt = np.zeros([w,h,slc],dtype=np.float32)
    for i in range(slc):
        opt[:,:,i] = rot(v[:,:,i],'cw')
    return opt

def dice(im1, im2):
    """
    Computes the Dice coefficient
    ----------
    im1 : boolean array
    im2 : boolean array
    If they are not boolean, they will be converted.
    -------
    It returns the Dice coefficient as a float on the range [0,1].
        1: Perfect overlapping 
        0: Not overlapping 
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.size != im2.size:
        raise ValueError("Size mismatch between input arrays!!!")
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 1.0
    # Compute Dice 
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum

def sim_stat(im,template):
    '''
    compute the TP, TN, FP, FN, sensitivity, specificity, accuracy
    '''
    diff = np.abs(im-template).astype(np.bool)
    im = np.asarray(im).astype(np.bool)
    template = np.asarray(template).astype(np.bool)
    if im.size != template.size:
        raise ValueError("Size mismatch between input arrays!!!")
    
    tp = np.sum(np.logical_and(im,template))
    tn = np.sum(np.logical_and(np.logical_not(im),np.logical_not(template)))
    fp = np.sum(np.logical_and(diff,np.logical_not(template)))
    fn = np.sum(np.logical_and(diff,template))
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    accuracy = (tn+tp)/(tn+tp+fn+fp)
    
    return sensitivity,1-specificity,accuracy

def ColorSeg(pred, gt):
    h, w = pred.shape
    tn_color = np.array(Image.new('RGB', (w, h), "seashell"))
    tp_color = np.array(Image.new('RGB', (w, h), "navy"))
    fp_color = np.array(Image.new('RGB', (w, h), "limegreen"))
    fn_color = np.array(Image.new('RGB', (w, h), "crimson"))
    
    true = np.uint8(pred == gt)
    tp_mask = true * gt
    tp_mask = np.repeat(tp_mask[:, :, np.newaxis], 3, axis=2)
    
    tn_mask = true * (1-gt)
    tn_mask = np.repeat(tn_mask[:, :, np.newaxis], 3, axis=2)
    
    fp_mask = np.uint8(pred > gt)
    fp_mask = np.repeat(fp_mask[:, :, np.newaxis], 3, axis=2)
    
    fn_mask = np.uint8(pred < gt)
    fn_mask = np.repeat(fn_mask[:, :, np.newaxis], 3, axis=2)
    
    im_color = tp_color * tp_mask + tn_color * tn_mask + \
               fp_color * fp_mask + fn_color * fn_mask
    
    return im_color
    
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)



