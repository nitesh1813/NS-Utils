#!/usr/bin/env python

# System
import os
import glob as gb
import pathlib as plib

# Third Party
import numpy as np

# Internal
import dvpy as dv
import segcnn

cg = segcnn.Experiment()
fs = segcnn.FileSystem(cg.base_dir, cg.data_dir)

def create_partition_lists():
  """
  The neural network is trained using cross-validation [1].  The dataset is divided into
  `n` "folds" or "partitions" (often 3 or 5).  Each of `n` models is trained on
  `n - 1` folds and tested on the remaining fold, and statistics are aggregated among
  the results from the five models.  This is a useful construction, especially when the
  amount of available data is limited.

  In our application (segmenting cardiac video sequences), we may have multiple ground
  truth annotations per patient.  Since the images from a single patient are highly correlated,
  it is not appropriate to have an image from a patient in the training set, and a different
  image from the same patient in the testing set.  For this reason, we partition the
  folds according to *patients*, rather than *images*.

  This function creates three lists:
  - partitions.npy: A list of patients, broken into `n` sublists.  For batch 0,
    partitions[0] contains the testing set and partitions[1:] contains the training set;
    likewise for all other batches.
  - img_list_*.npy: one per partition, individual images.
  - seg_list_*.npy: one per partition, individual segmentations.

  [1] https://en.wikipedia.org/wiki/Cross-validation_(statistics)
  """
  # Create the directory where these lists will be saved.
  os.makedirs(cg.data_dir, exist_ok = True)

  # Create a list of all patients.
  files = np.array(sorted(gb.glob(os.path.join(cg.base_dir, os.path.normpath('ucsd_*/*/'))+os.sep)))

  # Ensure that some patients are matched.
#  assert(len(files) > 0)

  # Shuffle the patients.
  np.random.shuffle(files)

  # Split the list into `cg.num_partitions` (approximately) equal sublists.
  partitions = np.array_split(files, cg.num_partitions)

  # Save the partitions.
  np.save(fs.partitions(), partitions)

def create_img_lists(imglist):

  partitions = np.load(fs.partitions())

  # Loop over the partitions.
  for i, partition in enumerate(partitions):

    ####################################
    ## Get list for all SEGMENTATIONS ##
    ####################################

    if imglist == 'ALL_SEGS':
      segs = [gb.glob(os.path.join(c, cg.seg_dir, fs.img('*'))) for c in partition]

      # Collapse (flatten) the nested list
      segs = dv.collapse_iterable(segs)

      # Build the image path corresponding to the given segmentation path.
      imgs = [os.path.join(os.path.dirname(os.path.dirname(s)), cg.img_dir, os.path.basename(s)) for s in segs]

    elif imglist == 'ALL_IMGS':

      # Sometimes we'll need to instead get a list for all IMAGES, rather than,
      # all SEGMENTATIONS, but this can be ignored for now.
      imgs = [gb.glob(os.path.join(c, cg.img_dir, fs.img('*'))) for c in partition]
      imgs = dv.collapse_iterable(imgs)
      segs = [os.path.join(os.path.dirname(os.path.dirname(i)), cg.seg_dir, os.path.basename(i)) for i in imgs]

    elif imglist == 'ED_ES':
      es = [os.path.join(c, 'es.txt') for c in partition]
      es = [int(open(s, 'r').read()) for s in es]
      segs = [[os.path.join(c, cg.seg_dir, fs.img(0)), os.path.join(c, cg.seg_dir, fs.img(f))] for c, f in zip(partition, es)]
      segs = dv.collapse_iterable(segs)
      imgs = [os.path.join(os.path.dirname(os.path.dirname(s)), cg.img_dir, os.path.basename(s)) for s in segs]
    else:
      print('Option not recognized.')
      assert(False)
      
    # Make sure that the two lists are the same length.
    assert(len(imgs) == len(segs))

    # Save the lists.
    os.makedirs(os.path.join(cg.data_dir, imglist), exist_ok = True)
    np.save(fs.img_list(i, imglist), imgs)
    np.save(fs.seg_list(i, imglist), segs)

if __name__ == '__main__':

  # Set a seed, so that np.random.shuffle() is reproducible.
  np.random.seed(cg.seed)

  # Create the partition lists.
  create_partition_lists()
  create_img_lists('ED_ES')
  create_img_lists('ALL_SEGS')
  create_img_lists('ALL_IMGS')

