import numpy as np
import nibabel as nib
import os
import gc

##########################      Normalization     ######################################


def normalize(x, axis=(0, 1, 2)):
    # normalize per volume (x,y,z) frame
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + 1e-8)

######################### Data loading ######################################
DATA_FOLDER = 'data/training'

volumes_nifti = [nib.load(os.path.join(DATA_FOLDER, f"patient{i:03d}.nii.gz")) for i in range(1, 101)]
segs_nifti = [nib.load(os.path.join(DATA_FOLDER, f"patient{i:03d}_seg.nii.gz")) for i in range(1, 101)]

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

del segs
gc.collect()

print(reslist[0].shape)
print(int(reslist[0][-1].squeeze()))
res = np.array([np.pad(res[..., 0], ((0,124),(0,127),(0,15)), constant_values=(1,))[..., np.newaxis] for res in reslist for t in range(1, int(res[-1].squeeze()))])
print(res[0].shape)

del reslist
gc.collect()

# save V0 elements as .npy files
for i, v in enumerate(V0):
    np.save(f"dataset/Vo{i:04d}.npy", v)

# save Vt elements as .npy files
for i, v in enumerate(Vt):
    np.save(f"dataset/Vt{i:04d}.npy", v)


# save M0 elements as .npy files
for i, m in enumerate(M0):
    np.save(f"dataset/Mo{i:04d}.npy", m)

# save Mt elements as .npy files
for i, m in enumerate(Mt):
    np.save(f"dataset/Mt{i:04d}.npy", m)

# save res elements as .npy files
for i, r in enumerate(res):
    np.save(f"dataset/res{i:04d}.npy", r)


