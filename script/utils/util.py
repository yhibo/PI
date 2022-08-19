"""
File: util.py

Author: Martin Nicolas
Version: 1.0
Description:
    Library for load and create hmd files. Also it is used to correrct the
    model coordiante system (paraview vs Slicer3d)

"""

from __future__ import print_function

import numpy as np
import SimpleITK as sitk
#  import matplotutils.pyplot as plt


class ScrollingWindows(object):
    def __init__(self, ax, data, ch=-1, slice_z=None, title='', **kwargs):
        self.ax = ax
        self.base_title = title
        self.slice_z = slice_z
        #  title = self.base_title + '/ use scroll wheel to navigate images'
        #  ax.set_title(title)

        self.data = data
        #  rows, cols, self.slices = data.shape
        self.ch = ch

        ind = data.shape[ch]//2
        self.ind =ind 
        self.update_slice()
        self.im = ax.imshow(self.data[self.slice], **kwargs)
        self.update()

    def onscroll(self, event):
        #  print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = np.clip(self.ind + 1, 0, self.data.shape[self.ch] - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.data.shape[self.ch] - 1)
        
        self.update_slice()
        self.update()
    
    def update_slice(self):
        if self.ch == 0 or self.ch == -3:
            self.slice  = np.s_[self.ind, :, :]
        elif self.ch == 1 or self.ch == -2:
            self.slice  = np.s_[:, self.ind, :]
        elif self.ch == 2 or self.ch == -1:
            self.slice  = np.s_[:, :, self.ind]

    def update(self):
        self.im.set_data(self.data[self.slice])
        if self.slice_z is None:
            title = self.base_title + ' - slice {}'.format(self.ind)
        else:
            title = self.base_title + ' - frame {}'.format(self.ind)
        self.ax.set_title(title)
        self.im.axes.figure.canvas.draw()

class ScrollingMotionWindows(ScrollingWindows):
    def __init__(self, ax, images, dfields, masks=None, ch=-1, title='',
            slice_z=None, frame=0, **kwargs):

        data = self.get_data(images, slice_z,frame, ch)
        motion = self.get_data(dfields, slice_z, frame, ch)
        mask = self.get_data(masks, slice_z, frame, ch)

        if mask.shape[0] == 1:
            mask = np.repeat(mask, data.shape[0], axis=0)


        self.mask = mask
        self.quiver = None

        if slice_z is None:
            title = '{} Frame {}'.format(title, frame)
        else:
            title = '{} Slice {}'.format(title, slice_z)

        super().__init__(ax, data,ch, slice_z, title, cmap=kwargs['cmap_img'])
        self.kwargs = kwargs
        self.kwargs.pop('cmap_img')

        self.ux = motion[...,0]
        self.uy = motion[...,1]
        if self.mask is not None:
            r, c = np.where(self.mask[self.slice]>0)
        else:
            r, c = np.where(self.data[self.slice]>0)
        self.quiver = ax.quiver(c, r, self.ux[self.slice][r,c],
                self.uy[self.slice][r,c], self.mask[self.slice][r,c], **kwargs)
        self.update()

    def update(self):
        super().update_slice()
        super().update()
        if self.mask is not None:
            r, c = np.where(self.mask[self.slice]>0)
        else:
            r, c = np.where(self.data[self.slice]>0)
        
        if self.quiver is not None:
            self.quiver.remove()
            self.quiver = self.ax.quiver(c, r, self.ux[self.slice][r,c],
                    self.uy[self.slice][r,c], self.mask[self.slice][r,c], **self.kwargs)
        self.im.axes.figure.canvas.draw()

    def get_data(self, idata, slice_z, frame, ch):
        '''
        Get the data according to the frames or slice_z along the time

        data: list of SimpleITK images where the coord. are [z,y,x]
        ret: Numpy Array
        '''

        if frame > len(idata):
            frame = len(idata)

        if slice_z is None and frame is not None:
            np_data = sitk.GetArrayFromImage(idata[frame])
        elif slice_z is not None:
            np_data = []
            for ii in range(len(idata)):
                np_data.append(sitk.GetArrayFromImage(idata[ii])[slice_z,...])

            np_data = np.array(np_data)
            
            if ch == 1 or ch == -2:
                np_data = np_data.swapaxes(1,0)
            elif ch == 2 or ch == -1:
                np_data = np_data.swapaxes(1,0).swapaxes(1,2)
        else:
            raise Exception('Error slice_z or frame must be set')

        return np_data


    



#  def multi_slice_viewer(data, fig, ax, ch=-1, **kwargs):
#      #  fig, ax = plt.subplots(1, 1)
#      if ch>2:
#          raise Exception('Error channel must be <3')
#      #  tracker = IndexTracker(ax, data, ch, **kwargs)
#      tracker = IndexTracker(ax, data, ch)
#      cid = fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
#      print(cid)
#      #  plt.show()



def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1,
                      length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def sitk2numpy(img):
    I = sitk.Cast(img, sitk.sitkUInt16)
    return sitk.GetArrayFromImage(I).round().astype(np.uint16)



#def physical2index(lmks_obs, orig, sp, id_vl, D):
#    # NOTE: Es mejor usar la funcion de ITK TransformPhysicalPointToIndex pero
#    # para eso cuando leo tengo que leer y guardar las img en simpleitk y luego
#    # cuando lo necesite convertirlas a numpy

#    def phy2idx(lmks_obsi, orig, sp, id_vl, D):
#       # 1: apex, 2: midventricular, 3: base
#       id_vl -= 1

#       # 4 lmks por nivel (apex, mid, basal)
#       i=4*id_vl
#       lmks_vl = lmks_obsi[i:i+4,:]

#       # Physical to index coordinates
#       # x = D.S.idx + o
#       S = np.diag(sp)
#       DS = np.matmul(D,S)
#       DSi = np.linalg.inv(DS)
#       idxi = np.matmul((lmks_vl - orig), DSi).round().astype(int)
#       return idxi

#    lmks_obs1 = lmks_obs[0,...]
#    idx1 = phy2idx(lmks_obs1, orig, sp, id_vl, D)
#    lmks_obs2 = lmks_obs[1,...]
#    idx2 = phy2idx(lmks_obs2, orig, sp, id_vl, D)
#    idx = np.array([idx1, idx2])

#    return idx


def cat_physical2index(img, lmks, id_vl=None):
    def phy2idx(img, lmksi, id_vl):
        # 1: apex, 2: midventricular, 3: base
        id_vl -= 1

        # 4 lmks por nivel (apex, mid, basal)
        i=4*id_vl
        lmks_vl = lmksi[i:i+4,:]
        idx = []

        for i in range(len(lmks_vl)):
            idx.append(img.TransformPhysicalPointToIndex(lmks_vl[i,:]))

        return np.array(idx)

    if id_vl is None:
        lmks_obs1 = lmks[0,...]
        lmks_obs2 = lmks[1,...]
        idx11 = phy2idx(img, lmks_obs1, 1)
        idx12 = phy2idx(img, lmks_obs1, 2)
        idx13 = phy2idx(img, lmks_obs1, 3)
        idx1 = np.vstack((idx11,idx12,idx13))

        idx21 = phy2idx(img, lmks_obs2, 1)
        idx22 = phy2idx(img, lmks_obs2, 2)
        idx23 = phy2idx(img, lmks_obs2, 3)
        idx2 = np.vstack((idx21,idx22,idx23))

        idx = np.array([idx1, idx2])
    else:
        lmks_obs1 = lmks[0,...]
        idx1 = phy2idx(img, lmks_obs1, id_vl)
        lmks_obs2 = lmks[1,...]
        idx2 = phy2idx(img, lmks_obs2, id_vl)
        idx = np.array([idx1, idx2])

    return idx




#def read_images(imag_dir): #, D, S):

#    filenames = [os.path.join(imag_dir, f) for f in os.listdir(imag_dir)
#                if os.path.isfile(os.path.join(imag_dir, f)) and
#                f.startswith('img_SA_') and f.endswith('.mhd')]

#    # Narutal sorting
#    filenamesMHD = natsorted(filenames)

#    return filenamesMHD

#def show_images(images, cols = 1, titles = None, xlim = None, ylim = None):

#    assert((titles is None) or (len(images) == len(titles)))
#    n_images = len(images)
#    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
#    fig = plt.figure()
#    for n, (image, title) in enumerate(zip(images, titles)):
#        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
#        if image.ndim == 2:
#            plt.gray()
#        plt.imshow(image)
#        if xlim is not None and ylim is not None:
#            plt.xlim([xlim[0], xlim[1]])
#            plt.ylim([ylim[1], ylim[0]])

#        a.set_title(title, fontsize = 8)
#    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
#    plt.show()
