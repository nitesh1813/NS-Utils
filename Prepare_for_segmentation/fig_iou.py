#!/usr/bin/env python

# System
import os
import pickle
from collections import OrderedDict

# Third Party
import numpy as np
import nibabel as nb
import pandas as pd
import pylab as plt
import seaborn as sb

# Internal
import dvpy as dv
import segcnn

# experiments = OrderedDict()

# for spacing in ['1-0', '1-5', '2-0']:
#   with open('./.experiments/left-heart-all-spacing-{}.p'.format(spacing), 'rb') as f:
#     experiments[spacing] = pickle.load(f)

# with open('./experiments/00-all-both-1-5-spacing.sh', 'rb') as f:
#   experiments['1-5'] = pickle.load(f)
# FIGURE_PATH=os.path.expandvars('${HOME}/Dropbox/datasets/valve-plane-detection-figures/')
# STAT_PATH = ""
experiment = segcnn.Experiment()
class_labels = {0 : 'Background',
                1 : 'LV',
                2 : 'LA',
                3 : 'LAA',
                4 : 'LVOT',
                5 : 'Ascending Aorta',
                6 : 'Left Inferior Pulmonary Vein',
                7 : 'Right Inferior Pulmonary Vein',
                8 : 'Left Superior Pulmonary Vein',
                9 : 'Right Superior Pulmonary Vein',
               }

def calculate_iou():

  cols = ['Spacing', 'Group', 'ID', 'Frame', 'Class', 'Signal', 'Noise', 'SNR', 'Jaccard']
  data = pd.DataFrame(columns = cols)
  spacing = '1-5'
  # for spacing, experiment in experiments.items():

    # print('Calculating Jaccard Index for spacing {}...'.format(spacing))

  fs = segcnn.FileSystem(experiment.base_dir, experiment.data_dir)

  imgs_list = [np.load(fs.img_list(p,'ALL_SEGS')) for p in range(experiment.num_partitions)]
  segs_list = [np.load(fs.seg_list(p,'ALL_SEGS')) for p in range(experiment.num_partitions)]
  imgs_list = np.concatenate(imgs_list)
  segs_list = np.concatenate(segs_list)

  for img, seg in zip(imgs_list, segs_list):

    gp = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(seg))))
    pt = os.path.basename(os.path.dirname(os.path.dirname(seg)))
    fm = int(os.path.splitext(os.path.splitext(os.path.basename(seg))[0])[0])

    x = nb.load(img).get_data()
    y_gt = nb.load(seg).get_data()
    y_pd = os.path.join(os.path.dirname(os.path.dirname(seg)),
                        'seg-pred-2',
                        os.path.basename(seg))
    print(y_pd)
    print(seg)
    y_pd = nb.load(y_pd).get_data()

    for c in range(1, experiment.num_classes):

      roi = x[y_gt == c]
      snr = np.nan
      signal = np.nan
      noise = np.nan
      if len(roi) != 0:
        signal = np.mean(roi)
        noise = np.std(roi)
        snr = signal / noise
      iou = dv.jaccard_index(y_gt, y_pd, c)
      dv.pandas_append_inplace(data, [spacing, gp, pt, fm, c, signal, noise, snr, iou])
    break

  data.sort_values(by = ['Spacing', 'Group', 'ID', 'Frame', 'Class'], inplace = True)

  data['Phase'] = np.where(data.Frame == 0, 'ED', 'ES')
  data['Manufacturer'] = np.where(data.Group == 'ucsd_siemens', 'Siemens', 'Toshiba')

  data.to_csv('stats.csv', index = False)

def plot_iou_vs_class():

  o_dir = os.path.join(FIGURE_PATH, '00_iou_vs_class')

  os.makedirs(o_dir, exist_ok = True)
  data = pd.read_csv(os.path.join(FIGURE_PATH, 'stats.csv'))

  for c, d in data.groupby('Spacing'):
    plt.suptitle('IOU vs Class (Spacing = {})'.format(c))

    sb.boxplot(x = 'Class', y = 'Jaccard', data = d, hue = 'Phase')
    sb.stripplot(x = 'Class', y = 'Jaccard', data = d, color = '0.3', dodge = True, jitter = True, hue = 'Phase')
    plt.gca().legend(loc='upper right')

    plt.ylim([0.0, 1.0])

    plt.savefig(os.path.join(o_dir, '{}.png'.format(c)))
    plt.close()

def plot_iou_vs_spacing():

  o_dir = os.path.join(FIGURE_PATH, '01_iou_vs_spacing')
  os.makedirs(o_dir, exist_ok = True)

  data = pd.read_csv(os.path.join(FIGURE_PATH, 'stats.csv'))

  for c, d in data.groupby('Class'):
    plt.suptitle('IOU vs Spacing / Depth ({})'.format(class_labels[c]))

    sb.stripplot(x = 'Spacing', y = 'Jaccard', data = d, dodge = True, jitter = True, hue = 'Phase')
    sb.pointplot(x = 'Spacing', y = 'Jaccard', data = d, hue = 'Phase')
#    sb.boxplot(x = 'Spacing', y = 'Jaccard', data = d, col = 'Phase')
    plt.gca().legend(loc='lower right')

    plt.ylim([0.0, 1.0])
    plt.savefig(os.path.join(o_dir, '{}.png'.format(c)))
    plt.close()

def plot_iou_vs_manufacturer():

  o_dir = os.path.join(FIGURE_PATH, '02_iou_vs_manufacturer')
  os.makedirs(o_dir, exist_ok = True)

  data = pd.read_csv(os.path.join(FIGURE_PATH, 'stats.csv'))
  data = data[data.Spacing == '1-5']

  for c, d in data.groupby('Class'):
    plt.suptitle('IOU vs Manufacturer ({})'.format(class_labels[c]))

    sb.boxplot(x = 'Manufacturer', y = 'Jaccard', data = d, hue = 'Phase')
    sb.stripplot(x = 'Manufacturer', y = 'Jaccard', data = d, color = '0.3', dodge = True, jitter = True, hue = 'Phase')
    plt.gca().legend(loc='lower right')

    plt.ylim([0.0, 1.0])
    plt.savefig(os.path.join(o_dir, '{}.png'.format(c)))
    plt.close()


def plot_snr_vs_manufacturer():

  o_dir = os.path.join(FIGURE_PATH, '03_snr_vs_manufacturer')
  os.makedirs(o_dir, exist_ok = True)

  data = pd.read_csv(os.path.join(FIGURE_PATH, 'stats.csv'))
  data = data[data.Spacing == '1-5']

  for c, d in data.groupby('Class'):
    plt.suptitle('SNR vs Manufacturer ({})'.format(class_labels[c]))

    sb.boxplot(x = 'Manufacturer', y = 'SNR', data = d, hue = 'Phase')
    sb.stripplot(x = 'Manufacturer', y = 'SNR', data = d, color = '0.3', dodge = True, jitter = True, hue = 'Phase')
    plt.gca().legend(loc='lower right')

    plt.ylim([0.0, None])
    plt.savefig(os.path.join(o_dir, '{}.png'.format(c)))
    plt.close()

def plot_iou_vs_snr():

  o_dir = os.path.join(FIGURE_PATH, '04_iou_vs_snr')
  os.makedirs(o_dir, exist_ok = True)

  data = pd.read_csv(os.path.join(FIGURE_PATH, 'stats.csv'))
  data = data[data.Spacing == '1-5']

  for c, d in data.groupby('Class'):
    plt.suptitle('IOU vs SNR ({})'.format(class_labels[c]))

    sb.jointplot(x = 'SNR', y = 'Jaccard', data = d, kind = 'reg')

    plt.ylim([0.0, 1.0])

    plt.savefig(os.path.join(o_dir, '{}.png'.format(c)))
    plt.close()

def plot_signal_vs_class():

  data = pd.read_csv(os.path.join(FIGURE_PATH, 'stats.csv'))
  data = data[data.Spacing == '1-5']

  sb.boxplot(y = 'SNR', x = 'Class', hue = 'Phase', data = data)

  plt.savefig(os.path.join(FIGURE_PATH, '05_signal_vs_class.png'))
  plt.close()

if __name__ == '__main__':

 calculate_iou()
#  plot_iou_vs_class()
#  plot_iou_vs_spacing()
#  plot_iou_vs_manufacturer()
#  plot_snr_vs_manufacturer()
#  plot_iou_vs_snr()
  # plot_signal_vs_class()
