import numpy as np
import nibabel as nib
import os
import gc
from natsort import natsorted

##########################      Normalization     ######################################


def normalize(x, axis=(0, 1, 2)):
    # normalize per volume (x,y,z) frame
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + 1e-8)

######################### Data loading ######################################
DATA_FOLDER = 'data/training'
patients = [f"patient{i:03d}" for i in range(1, 151)]


volumes_nifti = [nib.load(os.path.join(DATA_FOLDER, f"patient{i:03d}.nii.gz")) for i in range(1, 3)]
segs_nifti = [nib.load(os.path.join(DATA_FOLDER, f"patient{i:03d}_seg.nii.gz")) for i in range(1, 3)]

volumes = [normalize(v.get_fdata().astype('float32')) for v in volumes_nifti]
segs = [s.get_fdata().astype('float32')  for s in segs_nifti]
reslist = [np.array(v.header.get_zooms()[:-1]+(v.shape[-1],), dtype=np.float32).reshape(4,1,1,1) for v in volumes_nifti]

del volumes_nifti, segs_nifti
gc.collect()


V0 = np.array([v[..., 0][..., np.newaxis] for v in volumes for t in range(1, v.shape[-1])])
Vt = np.array([v[..., t][..., np.newaxis] for v in volumes for t in range(1, v.shape[-1])])

del volumes
gc.collect()

M0 = np.array([s[..., 0][..., np.newaxis] for s in segs for t in range(1, s.shape[-1])])
Mt = np.array([s[..., t][..., np.newaxis] for s in segs for t in range(1, s.shape[-1])])

print(V0.shape, Vt.shape, M0.shape, Mt.shape)

del segs
gc.collect()

print(reslist[0].shape)
print(int(reslist[0][-1].squeeze()))
res = np.array([np.pad(res[..., 0], ((0,124),(0,127),(0,15)), constant_values=(1,))[..., np.newaxis] for res in reslist for t in range(1, int(res[-1].squeeze()))])
print(res[0].shape)

del reslist
gc.collect()

######################### Data saving ######################################

# x is [V0, Vt] and y is np.array(V0, Vt, M0, Mt, res)

#save data in a TFRecord format
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(feature0, feature1, feature2, feature3, feature4):
    feature = {
        'V0': _bytes_feature(feature0),
        'Vt': _bytes_feature(feature1),
        'M0': _bytes_feature(feature2),
        'Mt': _bytes_feature(feature3),
        'res': _bytes_feature(feature4)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(data, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(len(data)):
            example = serialize_example(data[i][0].tobytes(), data[i][1].tobytes(), data[i][2].tobytes(), data[i][3].tobytes(), data[i][4].tobytes())
            writer.write(example)

data = list(zip(V0, Vt, M0, Mt, res))
write_tfrecord(data, 'data/training/training.tfrecord')




