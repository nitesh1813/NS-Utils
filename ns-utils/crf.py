import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral,create_pairwise_gaussian
import numpy as np
import sys
import tensorflow as tf
sys.path.append('../Prepare_for_segmentation/')
print(sys.path)


import segcnn
cg = segcnn.Experiment()

def Softmax(grid):
	with tf.device("gpu:0"):
		input = tf.placeholder('float32',shape=(cg.num_classes,)+cg.dim)
		softmax = tf.nn.softmax(input,axis=0)
		# sum = tf.reduce_sum(softmax)
		sess = tf.Session()
		grid = sess.run(softmax,feed_dict={input:grid})
	return grid

def crf(probs,image,shape,iterations):
	shape, NLABELS = cg.dim,cg.num_classes
	
	# segmentation = np.empty(shape)
	d = dcrf.DenseCRF(np.prod(shape), NLABELS)
	U = unary_from_softmax(probs)
	d.setUnaryEnergy(U)
	feats = create_pairwise_gaussian(sdims=(1.0, 1.0, 1.0), shape=shape)
	b_feats = create_pairwise_bilateral(sdims=(1.0, 1.0, 1.0),img=image,schan=(0.01,))
	d.addPairwiseEnergy(feats, compat=2, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	d.addPairwiseEnergy(b_feats,compat=3, kernel=dcrf.FULL_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	print("Starting Inference")
	Q = d.inference(iterations) 
	segmentation = np.argmax(Q, axis=0).reshape((shape[0], shape[1],shape[2]))

	return segmentation
