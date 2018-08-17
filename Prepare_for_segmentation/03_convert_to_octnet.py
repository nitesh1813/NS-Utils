import sys
import numpy as np
import os
from os.path import isfile, join
from collections import defaultdict

# Internal
import dvpy as dv
import segcnn

sys.path.append('../../py/')

import pyoctnet
# sys.path.append('../octnet/example/00_create_data/')
# import vis

import nibabel as nib
from nibabel.testing import data_path

#generates appropriate output path, 
thresholds = [175,225,200]
def output_path(input_path,ext,i):
	list = input_path.split("/")
	list[5] = "octreeDataset/AugmentedData/Data_160"+str(cg.spacing)+"_"+str(i)
	list[len(list)-1] = list[len(list)-1].split(".")[0]+ext
	output = "/".join(list[:len(list)-1])
	if not os.path.exists(output):
		os.makedirs(output)
	output = "/".join(list)
	return output
print(segcnn)
cg = segcnn.Experiment()
fs = segcnn.FileSystem(cg.base_dir, cg.data_dir)
print(cg.base_dir, cg.data_dir)

n_threads = 10

imgs_list_trn = np.load(fs.img_list(0, 'ED_ES'))

segs_list_trn = np.load(fs.seg_list(0, 'ED_ES')) 
assert(imgs_list_trn.shape == segs_list_trn.shape)
print(imgs_list_trn.shape)
clip=1000
output_img_list = defaultdict(list)
output_seg_list = defaultdict(list)
# # converted them into octrees and saves them as .oc files
def apply_affine_save(np_img,np_seg,i,img,seg,threshold):
	output_img = output_path(img,".oc",i)
	output_img_list[i].append(output_img)
	output_seg = output_path(seg,".oc",i)
	output_seg_list[i].append(output_seg)
	print(output_img)
	# if(isfile(output_img) and isfile(output_seg)):
	# 	return 0
# 	np_img,np_seg = random_transform(np_img,np_seg)

# 	np_seg +=1
# 	assert(np.amax(np_seg)<=11)
# # # 	print(np.amin(np_img))
# 	assert(np.amin(np_seg)>=1)
# 	# if(isfile(output_img) and isfile(output_seg)):
# 	# 	return 0
# 	print(np_seg.shape)
# 	np_img = np_img.copy(order = 'C').astype('float32')
# 	grid = pyoctnet.Octree.create_from_dense(np_img[np.newaxis], n_threads = n_threads,threshold = threshold)
# 	print(output_img)
# 	print(grid.grid_height(),grid.grid_depth(),grid.grid_width())
# 	print(grid.mem_using()/1024)
# 	byte_output = bytes(output_img, 'utf-8')
# 	grid.write_bin(byte_output)

# 	np_seg = np_seg.copy(order = 'C').astype('float32')
# 	grid = pyoctnet.Octree.create_from_dense(np_seg[np.newaxis], n_threads = n_threads,threshold = 0.9)
# 	# print(grid.grid_height(),grid.grid_depth(),grid.grid_width())
# 	print(output_seg)
# 	print(grid.mem_using()/1024)
# 	byte_output = bytes(output_seg, 'utf-8')
# 	grid.write_bin(byte_output)	
	return 0

# def converted_labels(np_seg,np_img):
# 	temp = np_seg.copy()
# 	bool_ = np_seg.copy()
# 	bool_[bool_>0] = 1
# 	bool_ = 1 - bool_
# 	# print(len(bool_[bool_==0]))
# 	# temp[np.where(np_img>500)] = 11
# 	temp[np.where(np_img<-500)] = 12
# 	temp[np.where(np_img==-2048)] = 11
# 	print(len(temp[temp==0]))
# 	temp = bool_*temp
# 	np_seg = np_seg + temp
# 	print(len(np_seg[np_seg==0]))
# 	# print(len(temp[temp==0]))
# 	# print(len(np_seg[np_seg>0]))
# 	# assert(len(np_seg[np_seg>0]) == len(temp[temp==0]))
# 	return np_seg

def random_transform( x, y):
	image_dimension = 2
	img_spatial_indices = np.array(range(0, image_dimension))
	translation_range = 0.1
	scale_range = 0.1
	img_channel_index = image_dimension
	cval=0
	fill_mode='constant'
	##
	## Translation
	##
	# print(y.ndim)
	translation = np.eye(image_dimension + 1)
	translation_range = (translation_range,) * image_dimension
	scale_range = (scale_range,) * image_dimension
	for t, ax in zip(translation_range, img_spatial_indices):
		translation[ax, image_dimension] = np.random.uniform(-t, t) * x.shape[ax - 1]
	# print(translation)
	##
	## Rotation
	##
	# print(translation.shape)
	rotation = np.eye(image_dimension + 1)

	theta = np.pi/4
	rotation[0,0] = np.cos(theta)
	rotation[0,1] = -np.sin(theta)
	rotation[1,0] = np.sin(theta)
	rotation[1,1] = np.cos(theta)

	##
	## Scale
	##

	scale = np.eye(image_dimension + 1)

	for s, ax in zip(scale_range, img_spatial_indices):
		scale[ax,ax] = np.random.uniform(1 - s, 1 + s)


	##
	## Compose and Apply
	##

	transform_matrix = np.dot(np.dot(rotation, translation), scale)
	# print(transform_matrix.shape,x.shape[:-1])
	print(transform_matrix[:-1,:].shape[0]==x.ndim)
	transform_matrix = dv.transform_full_matrix_offset_center(transform_matrix, x.shape[:-1])
	x = dv.apply_affine_transform_channelwise(x,
		  transform_matrix[:-1,:],
		  channel_index = 2,
		  fill_mode=fill_mode,
		  cval=cval,)
	print(transform_matrix[:-1,:].shape)
	# For y, mask data, fill mode constant, cval = 0
	y = dv.apply_affine_transform_channelwise(y,
		  transform_matrix[:-1,:],
		  channel_index = img_channel_index,
		  fill_mode = fill_mode,
		  cval= cval,)
	return x,y



for img,seg in zip(imgs_list_trn,segs_list_trn):
	print(img,seg)
	image = nib.load(img)
	segment = nib.load(seg)
	np_img = np.array(image.dataobj)
	
	

	np_seg = np.array(segment.dataobj)
	
	np_seg[np_seg>9]=10
	# print(len(np_seg[np_seg>0]),len(np_seg[np_seg==0]))


	# np_seg = converted_labels(np_seg,np_img)
	np_img[np_img<-clip] = -clip
	np_img[np_img>clip] = clip
	np_img = np_img
	np_img= dv.crop_or_pad(np_img, cg.dim)
	


	np_seg= dv.crop_or_pad(np_seg, cg.dim)
	# np_img,np_seg = random_transform(np_img,np_seg)
	# array_img = nib.Nifti1Image(np_img,image.affine,header=image.header)

	# nib.save(array_img,"test.nii.gz")
	# array_img = nib.Nifti1Image(np_seg,segment.affine,header=segment.header)

	# nib.save(array_img,"testseg.nii.gz")
	

	# np_seg[np_seg>10]=1
	
	# for threshold in thresholds:
		
		# break
	for i in range(50):
		apply_affine_save(np_img,np_seg,i,img,seg,175)
		# break
	# break


	# print("-------------------")]
# threshold = 175
# for image in imgs_list_trn:
# 	output = output_path(image,".oc",threshold)
# 	output_img_list.append(output)
# 	print(output)
# 	img =  nib.load(image)
# 	np_img = np.array(img.dataobj)
# 	# print("lololol")
# 	np_img[np_img<-clip] = -clip
# 	np_img[np_img>clip] = clip
# 	np_img= dv.crop_or_pad(np_img, (480,480,288))
# 	# print(cg.dim)

# 	np_img = np_img.copy(order = 'C').astype('float32')
# 	# print(np_img.nbytes/1024)
# 	# dense = np.zeros((480,480,288), dtype=np.float32)
# 	# dense = dense + 1

# 	# grid = pyoctnet.Octree.create_from_dense2(dense, np_img[np.newaxis,...].copy(), n_threads=n_threads,threshold = threshold)
# 	grid = pyoctnet.Octree.create_from_dense(np_img[np.newaxis], n_threads = n_threads,threshold = threshold)
# 	print(grid.grid_height(),grid.grid_depth(),grid.grid_width())
# 	# byte_output = bytes(output, 'utf-8')
# 	# grid.write_bin(byte_output)

# 	print(pyoctnet.Octree.mem_using(grid)/1024)
	
	

# for image in segs_list_trn:
# 	output = output_path(image,".oc")
# 	print(output)
# 	output_seg_list.append(output)
# 	img =  nib.load(image)
# 	np_img = np.array(img.dataobj)
# 	np_img = np_img
# 	# print(np.amax(np_img))
# 	# print(np.amin(np_img))
# 	np_img= dv.crop_or_pad(np_img, (384,384,192))
# 	np_img = np_img.copy(order = 'C').astype('float32')
# 	np_img +=1
# 	np_img[np_img>12]=1
# 	assert(np.amax(np_img)<=12)
# 	print(np.amin(np_img))
# 	assert(np.amin(np_img)>=1)
# 	grid = pyoctnet.Octree.create_from_dense(np_img[np.newaxis], n_threads = n_threads,threshold = 0.0000000000000000000001)
# 	# dense = np.zeros((480,480,288), dtype=np.float32)
# 	# dense = dense + 1

# 	# grid = pyoctnet.Octree.create_from_dense2(dense, np_img[np.newaxis,...].copy(), n_threads=n_threads,threshold = 0.0000000001)
# 	# grid = pyoctnet.Octree.create_from_dense(np_img[np.newaxis], n_threads = n_threads,threshold = threshold)
# 	print(grid.grid_height(),grid.grid_depth(),grid.grid_width())
# 	byte_output = bytes(output, 'utf-8')
# 	grid.write_bin(byte_output)
# 	# dense = grid.to_cdhw()
# 	# n = np.product(dense.shape)
# 	# for c in range(0, 12):
		
# 	# 	print(c,np.sum(dense==c)/n)
# 	# print(grid.grid_height(),grid.grid_depth(),grid.grid_width())
# 	# dense = grid.to_cdhw()
# 	# # print(np.amax(dense))
# 	# # print(dense.shape)
# 	# # break
# 	# # grid.write_dense()
# 	# byte_output = bytes(output, 'utf-8')
# 	print(pyoctnet.Octree.mem_using(grid)/1024)
# 	# break
# 	# grid.write_to_cdhw(byte_output)
	
# 	grid.write_bin(byte_output)
# 	break

# output_img_list = [inner  for outer in output_img_list.values()  for inner in outer]
# output_seg_list = [inner  for outer in output_seg_list.values()  for inner in outer]
# print(len(output_img_list))
# output_img_list = np.array( output_img_list)
# output_seg_list = np.array (output_seg_list)


# np.savetxt("/media/McVeighLab/projects/SNitesh/datasetsall-classes-all-phases-"+str(cg.spacing)+"/ED_ES/Aug_img_lists",output_img_list,fmt="%s")
# print("saved")
# np.savetxt("/media/McVeighLab/projects/SNitesh/datasetsall-classes-all-phases-"+str(cg.spacing)+"/ED_ES/Aug_seg_lists",output_seg_list,fmt="%s")
img_list = []
seg_list = []
# print(output_img_list.keys())
for k,v in output_img_list.items():
	np.savetxt("/media/McVeighLab/projects/SNitesh/datasetsall-classes-all-phases-"+str(cg.spacing)+"/ED_ES/Aug_img_lists_epoch_"+str(k),v,fmt="%s")
	img_list.append("/media/McVeighLab/projects/SNitesh/datasetsall-classes-all-phases-"+str(cg.spacing)+"/ED_ES/Aug_img_lists_epoch_"+str(k))
for k,v in output_seg_list.items():
	np.savetxt("/media/McVeighLab/projects/SNitesh/datasetsall-classes-all-phases-"+str(cg.spacing)+"/ED_ES/Aug_seg_lists_epoch_"+str(k),v,fmt="%s")
	seg_list.append("/media/McVeighLab/projects/SNitesh/datasetsall-classes-all-phases-"+str(cg.spacing)+"/ED_ES/Aug_seg_lists_epoch_"+str(k))

img_list = np.array(img_list)
seg_list = np.array(seg_list)
print(img_list.shape)
np.savetxt("/media/McVeighLab/projects/SNitesh/datasetsall-classes-all-phases-"+str(cg.spacing)+"/ED_ES/Aug_img_lists",img_list,fmt="%s")
print("saved")
np.savetxt("/media/McVeighLab/projects/SNitesh/datasetsall-classes-all-phases-"+str(cg.spacing)+"/ED_ES/Aug_seg_lists",seg_list,fmt="%s")