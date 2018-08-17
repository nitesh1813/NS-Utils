import nibabel as nib
import sys
import numpy as np
import os
from os.path import isfile, join
import dvpy as dv
import string
import re
def output_path(input_path,folder,data_folder,ext):
	list = input_path.split("/")
	list[5] = "octreeDataset/"+data_folder
	list[-1] = list[-1].split(".")[0]+ext
	list[-2] = folder
	output = "/".join(list[:len(list)-1])
	if not os.path.exists(output):
		os.makedirs(output)
	output = "/".join(list)
	return output

def convert_nii(y_pred,img):
	
	y_gt_nii=  nib.load(img)
	np_img = y_gt_nii.get_data()
	print(y_pred.shape)

	# print(np_img.shape)
	# y_pred = np.argmax(y_pred, axis = 0).astype(np.uint8)
	y_pred = dv.crop_or_pad(y_pred, y_gt_nii.get_data().shape)
	img_pred = nib.Nifti1Image(y_pred, y_gt_nii.affine,header=y_gt_nii.header)
	
	out = os.path.dirname(img)
	out = os.path.dirname(img[:-1])
	# print(out)
	out = os.path.join(os.path.join(os.path.dirname(out),"seg-pred-2"),os.path.basename(img))
	print(out)
	os.makedirs(os.path.dirname(out), exist_ok = True)
	# out = ("{0}.nii.gz".format(j))
	nib.save(img_pred, out)
	n = np.product(y_gt_nii.get_data().shape)
	for c in range(0, 12):
		print("Predicted = ",np.sum(y_pred==c)/n)
		print(c,np.sum(y_gt_nii.get_data()==c)/n)
		iou = dv.jaccard_index(y_gt_nii.get_data(), y_pred, c)
		print(iou)

# def get_seg_file(img):
# 	list = img.split("/")
def save_nii(img_grid,segmented_grid,path):
	
	input_nii = output_path(img,"img-nii-sm-crf",".nii.gz")
	affine = np.eye(4)
	img_pred = nib.Nifti1Image(img_grid, affine)
	for i in range(11):
		seg_nii = output_path(img,"seg_pred_nii",'_'+str(i)+".nii.gz")
		print(seg_nii)
		seg_pred = nib.Nifti1Image(segmented_grid[i],affine)
		nib.save(img_pred,input_nii)
		nib.save(seg_pred,seg_nii)