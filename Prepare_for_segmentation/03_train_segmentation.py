#!/usr/bin/env python

# System
import argparse
import os

# Third Party
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,Callback
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.utils import print_summary
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
## https://github.com/avolkov1/keras_experiments
# from keras_exp.multigpu import get_available_gpus, make_parallel
from keras.models import load_model
# Internal
import unet
from segcnn.generator import ImageDataGenerator
import segcnn.utils as ut
import dvpy as dv
import dvpy.tf
import segcnn
class MyCbk(Callback):

    def __init__(self, model,batch):

         self.model_to_save = model
         self.batch = batch

    def on_epoch_end(self, epoch, logs=None):
        print("saving model " + fs.model(self.batch))
        self.model_to_save.save(fs.model(self.batch))


# import os
os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
cg = segcnn.Experiment()
fs = segcnn.FileSystem(cg.base_dir, cg.data_dir)

K.set_image_dim_ordering('tf')  # Tensorflow dimension ordering in this code
# K.set_floatx('float16')
print(K.floatx())

os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
tf.logging.set_verbosity(tf.logging.INFO)
# Allow Dynamic memory allocation.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# cg.dim=(192,192,128)
# config.gpu_options.per_process_gpu_memory_fraction = 0.43
cg.batch_size= 2
#5375mb per batch
session = tf.Session(config=config)

def train(batch):

    #===========================================
    dv.section_print('Calculating Image Lists...')

    imgs_list_trn=[np.load(fs.img_list(p, 'ED_ES')) for p in range(cg.num_partitions)]
    segs_list_trn=[np.load(fs.seg_list(p, 'ED_ES')) for p in range(cg.num_partitions)]

    if batch is None:
      print('No batch was provided: training on all images.')
      batch = 'all'

      imgs_list_trn = np.concatenate(imgs_list_trn)
      segs_list_trn = np.concatenate(segs_list_trn)

      imgs_list_tst = imgs_list_trn
      segs_list_tst = segs_list_trn

    else:
      imgs_list_tst = imgs_list_trn.pop(batch)
      segs_list_tst = segs_list_trn.pop(batch)

      imgs_list_trn = np.concatenate(imgs_list_trn)
      segs_list_trn = np.concatenate(segs_list_trn)


    #===========================================
    dv.section_print('Creating and compiling model...')

    shape = cg.dim + (1,)
    # cg.batch_size = 1
    model_inputs = [Input(shape)]

    _, _, output = unet.get_unet(cg.dim,
                                    cg.num_classes,
                                    cg.conv_depth,
                                    0, # Stage
                                    dimension = len(cg.dim),
                                    unet_depth = cg.unet_depth,
                                   )(model_inputs[0])

    model_outputs = [output]
    with tf.device("/cpu:0"):
      models = Model(inputs = model_inputs,
                    outputs = model_outputs,
                   )

  #    # https://github.com/avolkov1/keras_experiments/blob/master/examples/mnist/mnist_tfrecord_mgpu.py
  #    model = make_parallel(model, get_available_gpus())

    print(cg.batch_size)
    # cbk = MyCbk(models,batch)
    # saved_model="/media/McVeighLab/projects/SNitesh/datasetsall-classes-all-phases-1.5/model_batch_all.hdf5"
    # if(os.path.isfile(saved_model)):
    #   models.load_weights(fs.model(batch), by_name=True)
    
    # model = multi_gpu_model(models, gpus=2)
    model = models
    opt = Adam(lr = 1e-3)
    model.compile(optimizer = opt,
                    loss = 'categorical_crossentropy')
    #===========================================
    dv.section_print('Fitting model...')

    # callbacks = [
    #              LearningRateScheduler(ut.step_decay),cbk
    #             ]
    callbacks = [ModelCheckpoint(fs.model(batch),
                                 monitor='val_loss',
                                 save_best_only=True,
                                 ),
                 LearningRateScheduler(ut.step_decay),
                ]
    # Training Generator
    datagen = ImageDataGenerator(
        3, # Dimension of input image
        translation_range = cg.xy_range,  # randomly shift images vertically (fraction of total height)
#        rotation_range = 0.0,  # randomly rotate images in the range (degrees, 0 to 180)
        scale_range = cg.zm_range,
        flip = cg.flip,
        )

    datagen_flow = datagen.flow(imgs_list_trn,
      segs_list_trn,
      batch_size = cg.batch_size,
      input_adapter = ut.in_adapt,
      output_adapter = ut.out_adapt,
      shape = cg.dim,
      input_channels = 1,
      output_channels = cg.num_classes,
      augment = True,
      )

    valgen = ImageDataGenerator(
        3, # Dimension of input image
        )
    print(cg.dim)

    valgen_flow = valgen.flow(imgs_list_tst,
      segs_list_tst,
      batch_size = cg.batch_size,
      input_adapter = ut.in_adapt,
      output_adapter = ut.out_adapt,
      shape = cg.dim,
      input_channels = 1,
      output_channels = cg.num_classes,
      )
    # file_write=open("model_description","w")
    # print_summary(model, line_length=None, positions=[.33, .75, .87, 1.], print_fn=None)
    # print(model.layers)
    # print(model.inputs)
    # print(model.outputs)
    # print(model.summary(),file=file_write)
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen_flow,
                        steps_per_epoch = imgs_list_trn.shape[0] //( cg.batch_size),
                        epochs = cg.epochs,
                        workers = 1,
                        validation_data = valgen_flow,
                        validation_steps = imgs_list_tst.shape[0] //(cg.batch_size),
                        callbacks = callbacks,
                        verbose = 1,
                       )
    print_summary(model, line_length=None, positions=None, print_fn=None)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int)
  args = parser.parse_args()

  if args.batch is not None:
    assert(0 <= args.batch < cg.num_partitions)

  train(args.batch)
