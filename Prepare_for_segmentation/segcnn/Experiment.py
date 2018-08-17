# System
import os

class Experiment():

  def __init__(self):
    # Input data (annotations).
    self.base_dir = os.environ['CG_INPUT_DIR']

    # Output data (models, image lists).
    self.data_dir = os.environ['CG_DERIVED_DIR']
  
    # Number of partitions in the crossvalidation.
    self.num_partitions = int(os.environ['CG_NUM_PARTITIONS'])
  
    # Dimension of padded input, for training.
    self.dim = (int(os.environ['CG_CROP_X']), int(os.environ['CG_CROP_Y']), int(os.environ['CG_CROP_Z']))
  
    # Seed for randomization.
    self.seed = int(os.environ['CG_SEED'])
  
    # Number of Classes (Including Background)
    self.num_classes = int(os.environ['CG_NUM_CLASSES'])
  
    # UNet Depth
    self.unet_depth = 5
  
    # Depth of convolutional feature maps
    self.conv_depth_multiplier = int(os.environ['CG_CONV_DEPTH_MULTIPLIER'])
    self.conv_depth = [16, 32, 64, 128, 256, 256, 128, 64, 32, 16, 16]
    self.conv_depth = [self.conv_depth_multiplier*x for x in self.conv_depth]
  
    assert(len(self.conv_depth) == (2*self.unet_depth+1))
  
    # How many images should be processed in each batch?
    self.batch_size = int(os.environ['CG_BATCH_SIZE'])
  
    # Translation Range
    self.xy_range = float(os.environ['CG_XY_RANGE'])
  
    # Scale Range
    self.zm_range = float(os.environ['CG_ZM_RANGE'])
    self.spacing= float(os.environ['CG_SPACING'])
    # Should Flip
    self.flip = False

    # Total number of epochs to train
    self.epochs = int(os.environ['CG_EPOCHS'])

    # Number of epochs to train before decreasing learning rate
    self.lr_epochs = int(os.environ['CG_LR_EPOCHS'])

    # Input directory names
    self.img_dir = os.path.normpath('img-nii-sm/')
    self.seg_dir = os.path.normpath('seg-nii-sm/')

    # Output directory name
    self.pred_dir = os.path.normpath('seg-pred-2')

