"""
File: io.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://github.com/curiale
Description: Funciones auxiliares de IO
"""

import os
import numpy as np
from tvtk.api import tvtk
import SimpleITK as sitk
from natsort import natsorted
#from config import CA_PHANTOM


CA_PHANTOM = "C:\\Balseiro\\Tesis - MFM\\Datos\\Tagging"

def dfield2vtp(Idfield, Iroi, origin, direction, spacing):
    z, y, x = np.where(Iroi)
    ## coord en paraview
    xm, ym, zm = x*spacing[0], y*spacing[1], z*spacing[2]
    # Add the origin
    xm += origin[0]
    ym += origin[1]
    zm += origin[2]

    dfield_pd = tvtk.PolyData()
    points = tvtk.Points()
    vertices = tvtk.CellArray()
    for ii in range(zm.shape[0]):
       pt = (xm[ii], ym[ii], zm[ii])
       pid = points.insert_next_point(pt)
       vertex = tvtk.Vertex()
       vertex.point_ids.set_id(0,pid)
       vertices.insert_next_cell(vertex)

    dfield_pd.points = points
    dfield_pd.verts = vertices

    if Idfield.shape[-1] < 3:
        Idisp = np.zeros(Idfield.shape[:3] + (3,))
        Idisp[...,0] = Idfield[..., 0]
        Idisp[...,1] = Idfield[..., 1]
    else:
        Idisp = Idfield

    disp = tvtk.FloatArray()
    disp.from_array(Idisp[z,y,x,:].flatten())
    disp.number_of_components=3
    disp.name = 'disp'

    dfield_pd.point_data.add_array(disp)
    return dfield_pd



def cat_read_harp_dfield(patient, numpy=True):
    path = os.path.join(CA_PHANTOM, patient, '3DTAG', 'Deformation', '3DTAG')
    fname = [os.path.join(path,f) for f in os.listdir(path) if
            f.startswith('dfield_2d_harp_') and f.endswith('.mhd')]
    fname = natsorted(fname)

    I = []
    for f in fname:
        img = sitk.ReadImage(f)
        if numpy:
            img = sitk.GetArrayFromImage(img)
        I.append(img)

    return I


def cat_read_aha_coord_pts(patient, dbtype='ssfp', morph_int=False):
    ''' Read the myocardial GT segmentation with the 17-AHA model information'''

    if dbtype.lower()=='ssfp':
        mpath = os.path.join(CA_PHANTOM, 'cMAC', 'GT', 'SSFP', patient, 'MESH',
                'VTK_COORDINATES')
    elif dbtype.lower()=='3dtag':
        mpath = os.path.join(CA_PHANTOM, 'cMAC', 'GT', '3DTAG', patient, 'MESH',
                'VTK_COORDINATES')

    suffix = ''
    if morph_int:
        suffix = '-mi'

    mname = os.path.join(mpath, 'aha_coord{}.vtp'.format(suffix))
    reader = tvtk.XMLPolyDataReader()
    reader.file_name = mname
    reader.update()
    mesh = reader.get_output()
    return mesh

def cat_read_aha_coord_mesh(patient, dbtype='ssfp'):
    ''' Read the myocardial GT segmentation with the 17-AHA model information'''

    if dbtype.lower()=='ssfp':
        mpath = os.path.join(CA_PHANTOM, 'cMAC', 'GT', 'SSFP', patient, 'MESH',
                'VTK_COORDINATES')
    elif dbtype.lower()=='3dtag':
        mpath = os.path.join(CA_PHANTOM, 'cMAC', 'GT', '3DTAG', patient, 'MESH',
                'VTK_COORDINATES')

    mname = os.path.join(mpath, 'mesh_aha_coord.vtp')
    reader = tvtk.XMLPolyDataReader()
    reader.file_name = mname
    reader.update()
    mesh = reader.get_output()
    return mesh


def cat_read_results_strain(patient, dbtype='upf'):

   lpath = os.path.join(CA_PHANTOM, 'cMAC', 'RESULTS', dbtype.upper(), 'SSFP',
           patient, 'MESH')


   fnames = [f for f in os.listdir(lpath) if
           f.startswith('finalMesh') and f.endswith('.vtk')]
   fnames = natsorted(fnames)

   meshes = []
   for t in range(len(fnames)):
       fname = os.path.join(lpath, fnames[t])
       reader = tvtk.PolyDataReader()
       reader.file_name = fname
       reader.update()
       meshes.append(reader.get_output())

   return meshes


def cat_read_mesh(patient):
    ''' Read the myocardial GT segmentation with the 17-AHA model information'''
    mpath = os.path.join(CA_PHANTOM, 'cMAC', 'GT',  '3DTAG', patient,
            'MESH', 'VTK_COORDINATES')
    mname = os.path.join(mpath, '{}.vtk'.format(patient))
    reader = tvtk.PolyDataReader()
    reader.file_name = mname
    reader.update()
    mesh = reader.get_output()
    return mesh

def cat_read_mylmks(patient, dbtype='3dtag', method='harp',
        tracking_3d=True, morph_int=False, use_roi=True, tracking_edges=True,
        use_t0=True):
    # TODO: implementar para 3dtag - HARP
    if dbtype.lower() == '3dtag':
        lpath = os.path.join(CA_PHANTOM, patient, '3DTAG', 'LMKS', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp':
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Deformation',
                'LMKS', 'VTK_COORDINATES')

    prefix = ''
    if  tracking_3d:
        prefix = prefix + '_3d'
    else:
        prefix = prefix + '_2d'

    if tracking_edges:
        prefix = prefix + '-edge'

    if use_roi:
        prefix = prefix + '-roi'

    if use_t0:
        prefix = prefix + '-t0'


    if  morph_int:
        prefix = prefix + '-mi_ft'
    else:
        prefix = prefix + '_ft'


    prefix = prefix + '_' + method.lower()


    frames = ['_'.join(f.split('_')[1:]) for f in os.listdir(lpath)
            if f.startswith('dlmks'+prefix) and f.endswith('.vtp')]
    frames = natsorted(frames)

    # [id_pt, x, y, z, frame_i]
    mylmks = np.zeros((12, 3, len(frames)))

    for idf in range(len(frames)):
        lname = os.path.join(lpath, 'dlmks_{}'.format(frames[idf]))
        reader = tvtk.XMLPolyDataReader()
        reader.file_name = lname
        reader.update()
        lmks = reader.get_output()

        #sort from apex to base
        lmks = lmks.points.to_array()

        mylmks[..., idf] = lmks

    return mylmks




def cat_write_mylmks(patient, my_lmks, dbtype='3dtag', method='harp',
        tracking_3d=True, morph_int=False, use_roi=True, tracking_edges=True,
        use_t0=True):
    if dbtype.lower() == '3dtag':
        lpath = os.path.join(CA_PHANTOM, patient, '3DTAG', 'LMKS', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp':
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Deformation',
                'LMKS', 'VTK_COORDINATES')

    if not os.path.lexists(lpath):
        os.makedirs(lpath)

    prefix = ''
    if  tracking_3d:
        prefix = prefix + '_3d'
    else:
        prefix = prefix + '_2d'

    if tracking_edges:
        prefix = prefix + '-edge'

    if use_roi:
        prefix = prefix + '-roi'

    if use_t0:
        prefix = prefix + '-t0'


    if  morph_int:
        prefix = prefix + '-mi_ft'
    else:
        prefix = prefix + '_ft'

    prefix1 = 'dlmks' + prefix

    for t in range(my_lmks.shape[-1]):
        poly = tvtk.PolyData()
        pts = tvtk.Points()
        pts.data = my_lmks[...,t]
        poly.points = pts

        fname = os.path.join(lpath,
                '{}_{}_motion_{:0=3d}.vtp'.format(prefix1, method, t))
        writer = tvtk.XMLPolyDataWriter()
        writer.file_name = fname
        writer.set_input_data(poly)
        writer.write()







def cat_read_lmks(patient, ba_channel_i, ap2ba_i, dbtype='3dtag', coord='vtk',
        sort=True):
    '''Read the GT Landmarks used for motion and strain evaluation.
    8 landmarks for phantom and 12 per volunteer (4x3): one landmark per wall
    (anterior, lateral, posterior and septal) per ventricula level (basal
    midventricular and apical).
    '''
    if coord.lower() == 'vtk':
        coord = 'VTK_COORDINATES'
    elif coord.lower() == 'dicom':
        coord = 'DICOM_COORDINATES'
    elif coord.lower() == 'inria':
        coord = 'INRIA_COORDINATES'
    


    if dbtype.lower() == '3dtag':
        lpath = os.path.join(CA_PHANTOM, 'cMAC', 'GT',  '3DTAG', patient,
                'LMKS', coord)
    elif dbtype.lower() == 'mevis_3dtag':
        lpath = os.path.join(CA_PHANTOM, 'cMAC', 'RESULTS',  'MEVIS', '3DTAG', 
                patient, 'LMKS')
    elif dbtype.lower() == 'iucl_3dtag':
        lpath = os.path.join(CA_PHANTOM, 'cMAC', 'RESULTS',  'IUCL', '3DTAG', 
                patient, 'LMKS')
    elif dbtype.lower() == 'upf_3dtag':
        lpath = os.path.join(CA_PHANTOM, 'cMAC', 'RESULTS',  'UPF', '3DTAG', 
                patient, 'LMKS')
    elif dbtype.lower() == 'inria_3dtag':
        lpath = os.path.join(CA_PHANTOM, 'cMAC', 'RESULTS',  'INRIA', '3DTAG', 
                patient, 'LMKS')
    elif dbtype.lower() == 'ssfp':
        lpath = os.path.join(CA_PHANTOM, 'cMAC', 'GT',  'SSFP', patient,
                'LMKS', coord)
    elif dbtype.lower() == 'ssfp_fix':
        lpath = os.path.join(CA_PHANTOM, 'cMAC', 'GT',  'SSFP', patient,
                'LMKS_fix', coord)
    elif dbtype.lower() == 'upf_ssfp':
        lpath = os.path.join(CA_PHANTOM, 'cMAC', 'RESULTS', 'UPF',  'SSFP', patient,
                'LMKS')
    elif dbtype.lower() == 'inria_ssfp':
      lpath = os.path.join(CA_PHANTOM, 'cMAC', 'RESULTS', 'INRIA',  'SSFP', patient,
              'LMKS')

    #  print(lpath)

    frames = [f.split('_')[1] for f in os.listdir(lpath) if f.startswith('obs1') and
            f.endswith('.vtk')]
    frames = natsorted(frames)

    # [obs, id_pt, x, y, z, frame_i]
    lmks = np.zeros((2, 12, 3, len(frames)))

    if sort:
        lname = os.path.join(lpath, 'obs1_{}'.format(frames[0]))
        reader = tvtk.PolyDataReader()
        reader.file_name = lname
        reader.update()
        lmks1 = reader.get_output()
        lmks1 = lmks1.points.to_array()

        if ap2ba_i == 0:
            idsort = np.argsort(lmks1[:, ba_channel_i])[::-1] # From apex to base
        else:
            idsort = np.argsort(lmks1[:, ba_channel_i]) # From apex to base


    else:
        idsort = np.r_[:12]


    for idf in range(len(frames)):
        lname = os.path.join(lpath, 'obs1_{}'.format(frames[idf]))
        reader = tvtk.PolyDataReader()
        reader.file_name = lname
        reader.update()
        lmks1 = reader.get_output()

        lname = os.path.join(lpath, 'obs2_{}'.format(frames[idf]))
        reader = tvtk.PolyDataReader()
        reader.file_name = lname
        reader.update()
        lmks2 = reader.get_output()

        #sort from apex to base
        lmks1 = lmks1.points.to_array()
        lmks2 = lmks2.points.to_array()

        lmks1 = lmks1[idsort,:]
        lmks2 = lmks2[idsort,:]

        lmks[0,..., idf] = lmks1
        lmks[1,..., idf] = lmks2

    return lmks

def cat_read_harp_img(patient, numpy=True):
    path = os.path.join(CA_PHANTOM, patient, '3DTAG', 'HARP_images')
    fname = [os.path.join(path,f) for f in os.listdir(path) if
            f.startswith('harp_phase_') and f.endswith('.mhd')]
    fname = natsorted(fname)

    I = []
    for f in fname:
        img = sitk.ReadImage(f)
        if numpy:
            img = sitk.GetArrayFromImage(img)
        I.append(img)

    return I


def cat_read_harp_mag_img(patient, numpy=True):
    path = os.path.join(CA_PHANTOM, patient, '3DTAG', 'HARP_images')
    fname = [os.path.join(path,f) for f in os.listdir(path) if
            f.startswith('harp_mag_') and f.endswith('.mhd')]
    fname = natsorted(fname)

    I = []
    for f in fname:
        img = sitk.ReadImage(f)
        if numpy:
            img = sitk.GetArrayFromImage(img)
        I.append(img)

    return I


def cat_read_image(patient):
    path = os.path.join(CA_PHANTOM, patient, '3DTAG', 'VTK')
    fname = [os.path.join(path,f) for f in os.listdir(path) if
            f.startswith('VTK') and f.endswith('.vtk')]
    fname = natsorted(fname)

    I = []
    for f in fname:
        img = sitk.ReadImage(f)
        I.append(img)

    return I

def cat_read_sfftp_image(patient,morph_int=False):
    path = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'cSAX')

    sufix = ''
    if morph_int:
        sufix = '-mi'

    fname = [os.path.join(path,f) for f in os.listdir(path) if
            f.startswith('{}_cSAX{}_time_'.format(patient, sufix))
            and  f.endswith('.mhd')]
    fname = natsorted(fname)
    I = []
    for f in fname:
        img = sitk.ReadImage(f)
        I.append(img)

    return I


def cat_read_dfield(patient, method='demons', tracking_3d=False,
        morph_int=False, use_roi=True, tracking_edges=True, use_t0=True):
    path = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Deformation', 'cSAX_New')

    prefix='dfield'
    if tracking_3d:
        prefix= prefix + '_3d'
    else:
        prefix= prefix + '_2d'

    #if tracking_edges:
    #    prefix = prefix + '-edge'

    if use_roi:
        prefix = prefix + '-roi'

    if use_t0:
        prefix = prefix + '-t0'

    if morph_int:
        prefix= prefix + '-mi'


    if method.lower() == 'demons':
        prefix = prefix + '_ft_demons_'
    elif method.lower() == 'bspline':
        prefix = prefix + '_ft_bspline_'


    fname = [os.path.join(path,f) for f in os.listdir(path) if
            f.startswith(prefix) and f.endswith('.mhd')]
    fname = natsorted(fname)
    I = []
    for f in fname:
        img = sitk.ReadImage(f)
        I.append(img)
    return I




def cat_write_meshes(patient, meshes, dbtype='3dtag', method='harp',
        tracking_3d=True, morph_int=False, use_roi=True, tracking_edges=True,
        use_t0=True):

    sufix = ''

    if dbtype.lower() == '3dtag':
        lpath = os.path.join(CA_PHANTOM, patient, '3DTAG', 'MESH', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp_mesh':
        sufix = '-smooth'
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Deformation',
                'MESH', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp_pts':
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Deformation',
                'POINTS', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp_lmks':
        sufix = '-mesh_smooth'
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Deformation',
                'LMKS', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp':
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Deformation',
                'MESH', 'VTK_COORDINATES')


    if not os.path.lexists(lpath):
        os.makedirs(lpath)

    prefix = '{}'.format(patient)
    if  tracking_3d:
        prefix = prefix + '_3d'
    else:
        prefix = prefix + '_2d'

    if tracking_edges:
        prefix = prefix + '-edge'

    if use_roi:
        prefix = prefix + '-roi'

    if use_t0:
        prefix = prefix + '-t0'


    if  morph_int:
        prefix = prefix + '-mi_ft'
    else:
        prefix = prefix + '_ft'

    prefix = prefix + sufix

    for t in range(len(meshes)):
        fname = os.path.join(lpath,
                '{}_{}_motion_{:0=3d}.vtp'.format(prefix, method, t))
        writer = tvtk.XMLPolyDataWriter()
        writer.file_name = fname
        writer.set_input_data(meshes[t])
        writer.write()




def cat_read_meshes(patient, dbtype='ssfp', method='bspline',
        tracking_3d=True, morph_int=False, use_roi=True, tracking_edges=True,
        use_t0=True):
    ''' Read the myocardial deformation mesh '''

    sufix = ''

    if dbtype.lower() == 'ssfp':
        mpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Deformation',
                    'MESH', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp_lmks':
        sufix = '-mesh_smooth'
        mpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Deformation',
                    'LMKS', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp_pts':
        mpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Deformation',
                    'POINTS', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp_mesh':
        sufix = '-smooth'
        mpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Deformation',
                    'MESH', 'VTK_COORDINATES')

    prefix = '{}'.format(patient)
    if  tracking_3d:
        prefix = prefix + '_3d'
    else:
        prefix = prefix + '_2d'

    if tracking_edges:
        prefix = prefix + '-edge'

    if use_roi:
        prefix = prefix + '-roi'

    if use_t0:
        prefix = prefix + '-t0'


    if  morph_int:
        prefix = prefix + '-mi_ft'
    else:
        prefix = prefix + '_ft'

    prefix = prefix + sufix + '_' + method

    names = [n for n in os.listdir(mpath) if n.startswith(prefix)]

    meshes = []
    for t in range(len(names)):
        fname = os.path.join(mpath,
                '{}_motion_{:0=3d}.vtp'.format(prefix, t))
        reader = tvtk.XMLPolyDataReader()
        reader.file_name = fname
        reader.update()
        mesh = reader.get_output()
        meshes.append(mesh)

    return meshes



def cat_write_strain(patient, meshes, dbtype='3dtag', method='harp',
        tracking_3d=True, morph_int=False, use_roi=True, tracking_edges=True,
        use_t0=True):

    sufix = ''

    if dbtype.lower() == '3dtag':
        lpath = os.path.join(CA_PHANTOM, patient, '3DTAG', 'MESH', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp_mesh':
        sufix = '-smooth'
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Strain',
                'MESH', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp':
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Strain',
                'MESH', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp_pts':
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Strain',
                'POINTS', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp_lmks':
        sufix = '-mesh_smooth'
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Strain',
                'LMKS', 'VTK_COORDINATES')

    if not os.path.lexists(lpath):
        os.makedirs(lpath)

    prefix = '{}'.format(patient)
    if  tracking_3d:
        prefix = prefix + '_3d'
    else:
        prefix = prefix + '_2d'

    if tracking_edges:
        prefix = prefix + '-edge'

    if use_roi:
        prefix = prefix + '-roi'

    if use_t0:
        prefix = prefix + '-t0'


    if  morph_int:
        prefix = prefix + '-mi_ft'
    else:
        prefix = prefix + '_ft'

    prefix = prefix + sufix

    for t in range(len(meshes)):
        fname = os.path.join(lpath,
                '{}_{}_strain_{:0=3d}.vtp'.format(prefix, method, t))
        writer = tvtk.XMLPolyDataWriter()
        writer.file_name = fname
        writer.set_input_data(meshes[t])
        writer.write()


def cat_read_strain(patient, dbtype='3dtag', method='harp',
        tracking_3d=True, morph_int=False, use_roi=True, tracking_edges=True,
        use_t0=True):

    sufix = ''

    if dbtype.lower() == '3dtag':
        lpath = os.path.join(CA_PHANTOM, patient, '3DTAG', 'MESH', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp':
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Strain',
                'MESH', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp_mesh':
        sufix = '-smooth'
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Strain',
               'MESH', 'VTK_COORDINATES')
    elif dbtype.lower() == 'ssfp_lmks':
        sufix = '-mesh_smooth'
        lpath = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Strain',
                'LMKS', 'VTK_COORDINATES')

    if not os.path.lexists(lpath):
        os.makedirs(lpath)

    prefix = '{}'.format(patient)
    if  tracking_3d:
        prefix = prefix + '_3d'
    else:
        prefix = prefix + '_2d'

    if tracking_edges:
        prefix = prefix + '-edge'

    if use_roi:
        prefix = prefix + '-roi'

    if use_t0:
        prefix = prefix + '-t0'


    if  morph_int:
        prefix = prefix + '-mi_ft'
    else:
        prefix = prefix + '_ft'

    prefix = prefix + sufix

    fnames = [f for f in os.listdir(lpath)
                if f.startswith('{}_{}_strain_'.format(prefix, method)) and
                f.endswith('.vtp')]
    n = len(fnames)

    meshes = []
    for t in range(n):
        fname = os.path.join(lpath,
                '{}_{}_strain_{:0=3d}.vtp'.format(prefix, method, t))
        reader = tvtk.XMLPolyDataReader()
        reader.file_name = fname
        reader.update()
        mesh = reader.get_output()
        meshes.append(mesh)

    return meshes

