"""
File: utils_aha.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://github.com/curiale
Description: funciones auxiliares para crear el modelo aha y el sistema de
coord. locales de la seg. que esta en MHD_Data/<patient>/Segmentation
"""

# import os
import numpy as np
from tvtk import array_handler
from tvtk.api import tvtk
import SimpleITK as sitk
from scipy import ndimage




def mesh2Image(mesh, spacing=[0.5, 0.5, 0.5], origin=None, dim=None,
        direction=None, outVal=0):
    """
    Usamos VTK para crear la imagen aha pero luego pasamos a ITK para poder
    almacenar la direccion de la misma.
    """

    # A) Extraemos la geometrica de la superficie
    surface = tvtk.DataSetSurfaceFilter()
    surface.set_input_data(mesh)
    surface.update()
    pd = surface.get_output()

    # B) Ahora convertimos el polyData a una vtkImageData
    whiteImage = tvtk.ImageData()

    if origin is None or dim is None:
        dim = np.zeros(3, dtype=int)
        bounds = pd.bounds

        origin = np.zeros(3)
        origin[0] = bounds[0] + spacing[0] / 2
        origin[1] = bounds[2] + spacing[1] / 2
        origin[2] = bounds[4] + spacing[2] / 2

        for i in range(3):
            dim[i] = int(np.ceil((bounds[i * 2 + 1] - bounds[i * 2]) /
                                 spacing[i]))

    whiteImage.spacing = spacing
    whiteImage.dimensions = dim
    whiteImage.extent = [0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1]
    whiteImage.origin =origin
    whiteImage.allocate_scalars(array_handler.get_vtk_array_type(np.uint8), 1)

    # Fill the image with foreground voxels
    nop = whiteImage.number_of_points

    I = np.ones(nop, dtype=np.uint8)
    whiteImage.point_data.get_array('ImageScalars').from_array(I)


    pol2stenc = tvtk.PolyDataToImageStencil()
    pol2stenc.set_input_data(pd)
    pol2stenc.output_origin = origin
    pol2stenc.output_spacing = spacing
    pol2stenc.output_whole_extent = whiteImage.extent
    pol2stenc.update()

    # cut the corresponding white image and set the background:
    imgstenc = tvtk.ImageStencil()
    imgstenc.set_input_data(whiteImage)
    imgstenc.set_stencil_connection(pol2stenc.output_port)
    imgstenc.reverse_stencil = 0
    imgstenc.background_value = outVal
    imgstenc.update()

    vtkimg = imgstenc.get_output()


    # VTK to ITK
    I = vtkimg.point_data.get_array('ImageScalars').to_array()

    # dimension esta en [x,y,z] pero los datos estan en [z,y,x]
    I.shape = dim[::-1]

    itkimg = sitk.GetImageFromArray(I)
    itkimg.SetSpacing(spacing)
    itkimg.SetOrigin(origin)
    itkimg.SetDirection(direction)


    return itkimg


def myo_from_mesh(mesh, sp, orig, dim, direction, ov=0):

    return mesh2Image(mesh, spacing=sp, origin=orig, dim=dim,
        direction=direction, outVal=ov)

def epi_from_mesh(mesh, sp, orig, dim, direction, ov=0):


    delaunay_filter = tvtk.Delaunay3D()
    delaunay_filter.set_input_data(mesh)
    delaunay_filter.update()
    epi_mesh = delaunay_filter.get_output()


    return mesh2Image(epi_mesh, spacing=sp, origin=orig, dim=dim,
        direction=direction, outVal=ov)




def lv_center(Imyo, ba_channel, a2b, by_slice):
   # NOTE:  IMPORTANTE !!!! El apex-base en el vol. se mueve en ba_channel
   center_lv = []
   if ba_channel == 2:
       mid =int(round(Imyo.shape[2]/2))
       cm = ndimage.measurements.center_of_mass(Imyo[..., mid])

       for x in range(Imyo.shape[2]-1, -1, -1):
           npixels = Imyo[..., x].sum()
           if npixels >0:
               if by_slice:
                   cm = ndimage.measurements.center_of_mass(Imyo[..., x])
               center_lv.append(cm + (x,))

   elif ba_channel == 1:
       mid =int(round(Imyo.shape[1]/2))
       cm = ndimage.measurements.center_of_mass(Imyo[:, mid, :])

       for y in range(Imyo.shape[1]-1, -1, -1):
           npixels = Imyo[:,y, :].sum()
           if npixels >0:
               if by_slice:
                   cm = ndimage.measurements.center_of_mass(Imyo[:, y ,:])
               center_lv.append((cm[0], y, cm[1]))

   elif ba_channel == 0:
       mid =int(round(Imyo.shape[0]/2))
       cm = ndimage.measurements.center_of_mass(Imyo[mid, ...])

       for z in range(Imyo.shape[0]-1, -1, -1):
           npixels = Imyo[z, ...].sum()
           if npixels >0:
               if by_slice:
                   cm = ndimage.measurements.center_of_mass(Imyo[z, ...])
               center_lv.append((z,) + cm)

   center_lv = np.array(center_lv)

   if a2b:
       center_lv = center_lv[::-1,:]
   return center_lv


def create_lv_coord2mesh(mesh, ba_channel, a2b):
    # NOTE: Se que el corazon siempre esta parado en el z appex_z<base_z
    # NOTE: El corazon esta oreintado en ba_channel con orientacion a2b. Si
    # a2b=0 entonces el cero esta mas cerca del apex que de la base. Si esta
    # a2b=1 entonces la base esta mas cerca del 0 que el apex

    # Coord radial Apuntando al interior
    normals = tvtk.PolyDataNormals()
    normals.set_input_data(mesh)
    normals.compute_cell_normals = False
    normals.compute_point_normals = True
    normals.auto_orient_normals = True
    normals.consistency = True
    normals.feature_angle = 30
    normals.splitting = True
    normals.update()
    lv_coords = normals.get_output()
    c_r = np.array(lv_coords.point_data.get_array('Normals'))

    # La normal apunta al exterior, por eso pasamos aquellas que apuntan hacia
    # afuera del dentro de cm hacia el centro de masa
    cm = tvtk.CenterOfMass()
    cm.set_input_data(mesh)
    cm.update()
    # En teoria no anda cm.center asi lque lo tomo del obj vtk
    center_lv = np.array(tvtk.to_vtk(cm).GetCenter())

    xyz = np.array(lv_coords.points)

    dist1 = np.linalg.norm((xyz + c_r) - center_lv, axis=1)
    dist2 = np.linalg.norm((xyz - c_r) - center_lv, axis=1)

    idflip = dist1>dist2

    c_r[idflip,:] = -c_r[idflip,:]

    plane = tvtk.Plane()

    npts = lv_coords.number_of_points
    c_l = np.zeros((npts,3))
    c_c = np.zeros((npts,3))


    if ba_channel == 0:
        if a2b:
            v_l = [0,0,-1]
        else:
            v_l = [0,0,1]
    elif ba_channel == 1:
        if a2b:
            v_l = [0,-1,0]
        else:
            v_l = [0,1,0]
    elif ba_channel == 2:
        if a2b:
            v_l = [-1,0,0]
        else:
            v_l = [1,0,0]

    for idp in range(npts):
        # Coord. circnferencial
        plane.origin = lv_coords.points[idp]
        plane.normal = c_r[idp,:]
        # Project a vector v onto plane defined by origin and normal.
        plane.project_vector(v_l, c_l[idp,:])


        # Coord. circnferencial
        c_c[idp,:] = np.cross(c_r[idp,:],c_l[idp,:])


    c_long = tvtk.FloatArray()
    c_long.from_array(c_l.flatten())
    c_long.number_of_components=3
    c_long.name = 'long'

    c_circ = tvtk.FloatArray()
    c_circ.from_array(c_c.flatten())
    c_circ.number_of_components=3
    c_circ.name = 'circ'

    c_rad = tvtk.FloatArray()
    c_rad.from_array(c_r.flatten())
    c_rad.number_of_components=3
    c_rad.name = 'rad'

    #lv_coords.point_data.add_array(aha)
    lv_coords.point_data.add_array(c_long)
    lv_coords.point_data.add_array(c_circ)
    lv_coords.point_data.add_array(c_rad)

    return lv_coords





def local_coord(Imyo, center_lv, x, y, z, ba_channel, a2b):

   # NOTE
   # Vamos a armar las coordenadas locales. De forma simplificada la coord
   # longitudinal es la que punta hacia la base desde el apex. Las
   # circunferencial y radial la sacamos de las coord. polares.
   coord = np.zeros(Imyo.shape + (3,3), dtype=float) # 3x3 = [c_l,c_c,c_r]
   # NOTE: recordar que tenemos cada coord en [z,y,x] y vamos a ponerlas en
   # [x,y,z] como c_l. Además el apex-base se mueve en ba_channel

   if ba_channel == 2:
       if a2b:
           c_l = [-1,0,0]
       else:
           c_l = [1,0,0]
       coord[z,y,x,:,0] = c_l
       for ii, x_i in enumerate(center_lv[:,2]):
           idz = x==x_i
           zy = np.vstack([z[idz],y[idz]]).T - center_lv[ii,:2]
           c_r = (zy.T/np.linalg.norm(zy, axis=1)).T
           # Coord radial Apuntando al interior
           c_r = -c_r
           # Coord. circnferencial
           c_c = np.vstack([c_r[:,1], -c_r[:,0]]).T

           # coord estan en [x,y,z] por eso invertimos c_c y c_r que los tenemos
           # en [y,x]
           coord[z[idz],y[idz],x[idz],1:,1] = c_c[:,::-1]
           coord[z[idz],y[idz],x[idz],1:,2] = c_r[:,::-1]

   elif ba_channel == 1:
       if a2b:
           c_l = [0,-1,0]
       else:
           c_l = [0,1,0]
       coord[z,y,x,:,0] = c_l
       for ii, y_i in enumerate(center_lv[:,1]):
           idy = y==y_i
           zx = np.vstack([z[idy],x[idy]]).T - center_lv[ii,::2] # centro de [z,x]
           c_r = (zx.T/np.linalg.norm(zx, axis=1)).T
           # Coord radial Apuntando al interior
           c_r = -c_r
           # Coord. circnferencial
           c_c = np.vstack([c_r[:,1], -c_r[:,0]]).T

           # coord estan en [x,y,z] por eso invertimos c_c y c_r que los tenemos
           # en [y,x]
           coord[z[idy],y[idy],x[idy],::2,1] = c_c[:,::-1]
           coord[z[idy],y[idy],x[idy],::2,2] = c_r[:,::-1]

   elif ba_channel == 0:
       if a2b:
           c_l = [0,0,-1]
       else:
           c_l = [0,0,1]
       coord[z,y,x,:,0] = c_l
       for ii, z_i in enumerate(center_lv[:,0]):
           idz = z==z_i
           yx = np.vstack([y[idz],x[idz]]).T - center_lv[ii,1:]
           c_r = (yx.T/np.linalg.norm(yx, axis=1)).T
           # Coord radial Apuntando al interior
           c_r = -c_r
           # Coord. circnferencial
           c_c = np.vstack([c_r[:,1], -c_r[:,0]]).T

           # coord estan en [x,y,z] por eso invertimos c_c y c_r que los tenemos
           # en [y,x]
           coord[z[idz],y[idz],x[idz],:2,1] = c_c[:,::-1]
           coord[z[idz],y[idz],x[idz],:2,2] = c_r[:,::-1]


   else:
       raise Exception('Uknown channel {}'.format(ba_channel))



   return coord



def lv_local_coord_system(myo, ba_channel, a2b, by_slice=True):
   # NOTE:  IMPORTANTE !!!! El apex-base en el vol. se mueve en ba_channel
   # y a2b indica si esta invertido, es decir mirando hacia abajo

   # dim = myo.GetSize()
   origin = myo.GetOrigin()
   spacing = myo.GetSpacing()
   direction = myo.GetDirection()

   Imyo = sitk.GetArrayFromImage(myo)
   center_lv = lv_center(Imyo, ba_channel, a2b, by_slice)

   # Vamos a cambiar el centro de los que son muy pequeños por otros donde hay
   # mas LV

   z, y, x = np.where(Imyo)

   coord = local_coord(Imyo, center_lv, x,y,z, ba_channel, a2b)

   coord_img = sitk.GetImageFromArray(coord.reshape(coord.shape[:3]+(9,)),
          isVector=True)
   coord_img.SetSpacing(spacing)
   coord_img.SetOrigin(origin)
   coord_img.SetDirection(direction)


   ## Creamos la malla en vtk con toda la informacion para visualizar las
   ## coord en paraview
   xm, ym, zm = x*spacing[0], y*spacing[1], z*spacing[2]
   # Add the origin
   xm += origin[0]
   ym += origin[1]
   zm += origin[2]

   coord_pd = tvtk.PolyData()
   points = tvtk.Points()
   vertices = tvtk.CellArray()
   for ii in range(zm.shape[0]):
       pt = (xm[ii], ym[ii], zm[ii])
       pid = points.insert_next_point(pt)
       vertex = tvtk.Vertex()
       vertex.point_ids.set_id(0,pid)
       vertices.insert_next_cell(vertex)

   coord_pd.points = points
   coord_pd.verts = vertices

   c_long = tvtk.FloatArray()
   c_long.from_array(coord[z,y,x,:,0].flatten())
   c_long.number_of_components=3
   c_long.name = 'long'

   c_rad = tvtk.FloatArray()
   c_rad.from_array(coord[z,y,x,:,1].flatten())
   c_rad.number_of_components=3
   c_rad.name = 'circ'


   c_circ = tvtk.FloatArray()
   c_circ.from_array(coord[z,y,x,:,2].flatten())
   c_circ.number_of_components=3
   c_circ.name = 'rad'

   coord_pd.point_data.add_array(c_long)
   coord_pd.point_data.add_array(c_rad)
   coord_pd.point_data.add_array(c_circ)

   return coord_img, coord_pd


def add_aha(mesh, rv_pt_phy, ba_channel, a2b):
    # NOTE: el corazon esta parado en la coord ba_channel parado como indica
    # a2b. Recordar que ba_channel indica el axis en formato [z,y,x]
    # NOTE: tomo el centro del bound en lugar del centro de masa
    bounds = np.array(mesh.bounds)
    bounds.shape=(3,2)

    center_lv = bounds[:,0] + (bounds[:,1] - bounds[:,0]) / 2


    rv2lv = center_lv - rv_pt_phy
    if ba_channel == 0:
        # Esta en z 
        rv2lv = rv2lv[:2]
    elif ba_channel == 1:
        # Esta en y 
        rv2lv = rv2lv[::2]
    else:
        # Esta en x
        rv2lv = rv2lv[1:]

    e1 = rv2lv/np.linalg.norm(rv2lv)

    # Vector ortonormal a e1
    e2 = np.array([e1[1], -e1[0]])

    tmatrix = np.matrix([e1, e2]).T

    npts = mesh.number_of_points
    aha_lv = np.zeros(npts, dtype=np.uint8)
    xyz = np.array(mesh.points)

    # Vamos a ignorar el segmento 17 porque es muy chico.
    if ba_channel == 0:
        # Esta en z 
        z_min = bounds[2,0]
        z_max = bounds[2,1]
        dz = z_max - z_min

        if a2b:
            # La appex esta en el maximo
            z_base = z_min + (.35*dz)
            z_midd = z_base + (.35*dz)
            id_base = xyz[:,2]<= z_base
            id_midd = (xyz[:,2]> z_base) & (xyz[:,2]<= z_midd)
            id_appex = xyz[:,2]> z_midd
        else:
            z_appex = z_min + (.3*dz)
            z_midd = z_appex + (.35*dz)

            id_appex = xyz[:,2]<= z_appex
            id_midd = (xyz[:,2]> z_appex) & (xyz[:,2]<= z_midd)
            id_base = xyz[:,2]> z_midd

        pts_appex = np.sum(id_appex)
        pts_midd = np.sum(id_midd)
        pts_base = np.sum(id_base)

        aha_lv_a = np.zeros(pts_appex, dtype=np.uint8)
        aha_lv_m = np.zeros(pts_midd, dtype=np.uint8)
        aha_lv_b = np.zeros(pts_base, dtype=np.uint8)


        # --- Base
        xy_base = xyz[id_base,:2] - center_lv[:2]
        XY_b = np.array(xy_base * tmatrix)
        ang_b = np.arctan2(XY_b[:,1], XY_b[:,0]) * 180 / np.pi

        # --- Midd-cavity
        xy_midd = xyz[id_midd,:2] - center_lv[:2]
        XY_m = np.array(xy_midd * tmatrix)
        ang_m = np.arctan2(XY_m[:,1], XY_m[:,0]) * 180 / np.pi
        
        # --- Appex
        xy_appex = xyz[id_appex,:2] - center_lv[:2]
        XY_a = np.array(xy_appex * tmatrix)
        ang_a = np.arctan2(XY_a[:,1], XY_a[:,0]) * 180 / np.pi

    elif ba_channel == 1:
        # Esta en y 
        y_min = bounds[1,0]
        y_max = bounds[1,1]
        dy = y_max - y_min

        if a2b:
            # La appex esta en el maximo
            y_base = y_min + (.35*dy)
            y_midd = y_base + (.35*dy)
            id_base = xyz[:,1]<= y_base
            id_midd = (xyz[:,1]> y_base) & (xyz[:,1]<= y_midd)
            id_appex = xyz[:,1]> y_midd
        else:
            y_appex = y_min + (.3*dy)
            y_midd = y_appex + (.35*dy)

            id_appex = xyz[:,1]<= y_appex
            id_midd = (xyz[:,1]> y_appex) & (xyz[:,1]<= y_midd)
            id_base = xyz[:,1]> y_midd

        pts_appex = np.sum(id_appex)
        pts_midd = np.sum(id_midd)
        pts_base = np.sum(id_base)

        aha_lv_a = np.zeros(pts_appex, dtype=np.uint8)
        aha_lv_m = np.zeros(pts_midd, dtype=np.uint8)
        aha_lv_b = np.zeros(pts_base, dtype=np.uint8)
        
        # --- Base
        xy_base = xyz[id_base,::2] - center_lv[::2]
        XY_b = np.array(xy_base * tmatrix)
        ang_b = np.arctan2(XY_b[:,1], XY_b[:,0]) * 180 / np.pi

        # --- Midd-cavity
        xy_midd = xyz[id_midd,::2] - center_lv[::2]
        XY_m = np.array(xy_midd * tmatrix)
        ang_m = np.arctan2(XY_m[:,1], XY_m[:,0]) * 180 / np.pi
        
        # --- Appex
        xy_appex = xyz[id_appex,::2] - center_lv[::2]
        XY_a = np.array(xy_appex * tmatrix)
        ang_a = np.arctan2(XY_a[:,1], XY_a[:,0]) * 180 / np.pi

    else:
        # Esta en x
        x_min = bounds[0,0]
        x_max = bounds[0,1]
        dx = x_max - x_min

        if a2b:
            # La appex esta en el maximo
            x_base = x_min + (.35*dx)
            x_midd = x_base + (.35*dx)
            id_base = xyz[:,0]<= x_base
            id_midd = (xyz[:,0]> x_base) & (xyz[:,0]<= x_midd)
            id_appex = xyz[:,0]> x_midd
        else:
            x_appex = x_min + (.3*dx)

        x_midd = x_appex + (.35*dx)

        id_appex = xyz[:,0]<= x_appex
        id_midd = (xyz[:,0]> x_appex) & (xyz[:,0]<= x_midd)
        id_base = xyz[:,0]> x_midd

        pts_appex = np.sum(id_appex)
        pts_midd = np.sum(id_midd)
        pts_base = np.sum(id_base)

        aha_lv_a = np.zeros(pts_appex, dtype=np.uint8)
        aha_lv_m = np.zeros(pts_midd, dtype=np.uint8)
        aha_lv_b = np.zeros(pts_base, dtype=np.uint8)
        
        # --- Base
        xy_base = xyz[id_base,1:] - center_lv[1:]
        XY_b = np.array(xy_base * tmatrix)
        ang_b = np.arctan2(XY_b[:,1], XY_b[:,0]) * 180 / np.pi

        # --- Midd-cavity
        xy_midd = xyz[id_midd,1:] - center_lv[1:]
        XY_m = np.array(xy_midd * tmatrix)
        ang_m = np.arctan2(XY_m[:,1], XY_m[:,0]) * 180 / np.pi
        
        # --- Appex
        xy_appex = xyz[id_appex,1:] - center_lv[1:]
        XY_a = np.array(xy_appex * tmatrix)
        ang_a = np.arctan2(XY_a[:,1], XY_a[:,0]) * 180 / np.pi
    
    # --- Base
    # Cambiamos el sistema de coordenadas
    # NOTE: recordar que visualmente en el sistema de coord el origen esta arriba,
    # asi que invertimos los numeros de los segmentos
    # ids_6 <-> 5, 1 <-> 4, 3 <->2
    ids_4 =  (60<= ang_b) & (ang_b <120)
    ids_3 =  (120<= ang_b) & (ang_b <180)
    ids_2 =  (-180<= ang_b) & (ang_b <-120)
    ids_1 =  (-120<= ang_b) & (ang_b <-60)
    ids_6 =  (-60<= ang_b) & (ang_b <0)
    ids_5 =  (0<= ang_b) & (ang_b <60)

    aha_lv_b[ids_1] = 1
    aha_lv_b[ids_2] = 2
    aha_lv_b[ids_3] = 3
    aha_lv_b[ids_4] = 4
    aha_lv_b[ids_5] = 5
    aha_lv_b[ids_6] = 6

    aha_lv[id_base] = aha_lv_b

    # --- Midd-cavity
    # Cambiamos el sistema de coordenadas
    # NOTE: recordar que visualmente en el sistema de coord el origen esta arriba,
    # asi que invertimos los numeros de los segmentos
    # ids_12 <-> 11, 7 <-> 10, 8 <->9
    ids_10 =  (60<= ang_m) & (ang_m <120)
    ids_9 =  (120<= ang_m) & (ang_m <180)
    ids_8 =  (-180<= ang_m) & (ang_m <-120)
    ids_7 =  (-120<= ang_m) & (ang_m <-60)
    ids_12 =  (-60<= ang_m) & (ang_m <0)
    ids_11 =  (0<= ang_m) & (ang_m <60)

    aha_lv_m[ids_7] = 7
    aha_lv_m[ids_8] = 8
    aha_lv_m[ids_9] = 9
    aha_lv_m[ids_10] = 10
    aha_lv_m[ids_11] = 11
    aha_lv_m[ids_12] = 12

    aha_lv[id_midd] = aha_lv_m

    # --- Appex
    # Cambiamos el sistema de coordenadas
    # NOTE: recordar que visualmente en el sistema de coord el origen esta arriba,
    # asi que invertimos los numeros de los segmentos
    # ids_13 <-> 15
    ids_16 =  (-45<= ang_a) & (ang_a <45)
    ids_15 =  (45<= ang_a) & (ang_a <135)
    ids_14_1 =  (135<= ang_a) & (ang_a <180)
    ids_14_2 =  (-180<= ang_a) & (ang_a <-135)
    ids_13 =  (-135<= ang_a) & (ang_a <-45)

    aha_lv_a[ids_13] = 13
    aha_lv_a[ids_14_1] = 14
    aha_lv_a[ids_14_2] = 14
    aha_lv_a[ids_15] = 15
    aha_lv_a[ids_16] = 16

    aha_lv[id_appex] = aha_lv_a

    aha = tvtk.UnsignedCharArray()
    aha.from_array(aha_lv.flatten())
    aha.number_of_components=1
    aha.name = 'aha'

    mesh.point_data.add_array(aha)



def create_aha(myo, rv_pt, ba_channel, a2b, by_slice=True):
   # NOTE:  IMPORTANTE !!!! El apex-base en el vol. se mueve en ba_channel

   Imyo = sitk.GetArrayFromImage(myo)
   center_lv = lv_center(Imyo, ba_channel, a2b, by_slice)

   if ba_channel==0:
       rv2lv = center_lv[0][1:] - rv_pt[1:]
   elif ba_channel==1:
       rv2lv = center_lv[0][::2] - rv_pt[::2]
   elif ba_channel==2:
       rv2lv = center_lv[0][:2] - rv_pt[:2]


   e1 = rv2lv/np.linalg.norm(rv2lv)


   # Vector ortonormal a e1
   e2 = np.array([e1[1], -e1[0]])

   tmatrix = np.matrix([e1, e2]).T

   # Vamos a ignorar el segmento 17 porque es muy chico.
   z_base = int(round(center_lv.shape[0] * .35))
   z_appex = int(round(center_lv.shape[0] * .3))
   z_mid = center_lv.shape[0] - z_base - z_appex


   aha_lv = np.zeros_like(Imyo, dtype=np.uint8)
   z, y, x = np.where(Imyo)
   # Creamos los segmentos acorde con la zona del LV

   if ba_channel ==0:
      # --- Base
       for idz, z_i in enumerate(center_lv[:z_base,ba_channel]):
           idx = (z == z_i)
           z_idx = z[idx]
           y_i = y[idx]
           x_i = x[idx]
           yx_i = (np.vstack([y_i, x_i]).T - center_lv[idz,1:]).T
           # Cambiamos el sistema de coordenadas
           # NOTE: recordar que visualmente en el sistema de coord el origen esta arriba,
           # asi que invertimos los numeros de los segmentos
           # ids_6 <-> 5, 1 <-> 4, 3 <->2
           YX_i = np.array(tmatrix * yx_i).T
           ang = np.arctan2(YX_i[:,1], YX_i[:,0]) * 180 / np.pi

           # Por una cuestion de redondeo voy a poner el = en todos ya que
           # pueden quedar agujeros sino se pone.
           ids_4 =  (60<= ang) & (ang <=120)
           ids_3 =  (120<= ang) & (ang <=180)
           ids_2 =  (-180<= ang) & (ang <=-120)
           ids_1 =  (-120<= ang) & (ang <=-60)
           ids_6 =  (-60<= ang) & (ang <=0)
           ids_5 =  (0<= ang) & (ang <=60)

           aha_lv[z_idx[ids_1], y_i[ids_1], x_i[ids_1]] = 1
           aha_lv[z_idx[ids_2], y_i[ids_2], x_i[ids_2]] = 2
           aha_lv[z_idx[ids_3], y_i[ids_3], x_i[ids_3]] = 3
           aha_lv[z_idx[ids_4], y_i[ids_4], x_i[ids_4]] = 4
           aha_lv[z_idx[ids_5], y_i[ids_5], x_i[ids_5]] = 5
           aha_lv[z_idx[ids_6], y_i[ids_6], x_i[ids_6]] = 6

       # --- Midd-cavity
       for idz, z_i in enumerate(center_lv[z_base:z_base+z_mid,ba_channel]):
           idx = (z == z_i)
           z_idx = z[idx]
           y_i = y[idx]
           x_i = x[idx]
           yx_i = (np.vstack([y_i, x_i]).T - center_lv[z_base+idz,1:]).T
           # Cambiamos el sistema de coordenadas
           # NOTE: recordar que visualmente en el sistema de coord el origen esta arriba,
           # asi que invertimos los numeros de los segmentos
           # ids_12 <-> 11, 7 <-> 10, 8 <->9
           YX_i = np.array(tmatrix * yx_i).T
           ang = np.arctan2(YX_i[:,1], YX_i[:,0]) * 180 / np.pi

           ids_10 =  (60<= ang) & (ang <=120)
           ids_9 =  (120<= ang) & (ang <=180)
           ids_8 =  (-180<= ang) & (ang <=-120)
           ids_7 =  (-120<= ang) & (ang <=-60)
           ids_12 =  (-60<= ang) & (ang <=0)
           ids_11 =  (0<= ang) & (ang <=60)

           aha_lv[z_idx[ids_7], y_i[ids_7], x_i[ids_7]] = 7
           aha_lv[z_idx[ids_8], y_i[ids_8], x_i[ids_8]] = 8
           aha_lv[z_idx[ids_9], y_i[ids_9], x_i[ids_9]] = 9
           aha_lv[z_idx[ids_10], y_i[ids_10], x_i[ids_10]] = 10
           aha_lv[z_idx[ids_11], y_i[ids_11], x_i[ids_11]] = 11
           aha_lv[z_idx[ids_12], y_i[ids_12], x_i[ids_12]] = 12


       # --- Appex NOTE: revisar que siempre el apex es el mas chico en z
       for idz, z_i in enumerate(center_lv[z_base+z_mid:,ba_channel]):
           idx = (z == z_i)
           z_idx = z[idx]
           y_i = y[idx]
           x_i = x[idx]
           yx_i = (np.vstack([y_i, x_i]).T - center_lv[z_base+z_mid+idz,1:]).T
           # Cambiamos el sistema de coordenadas
           # NOTE: recordar que visualmente en el sistema de coord el origen esta arriba,
           # asi que invertimos los numeros de los segmentos
           # ids_13 <-> 15
           YX_i = np.array(tmatrix * yx_i).T
           ang = np.arctan2(YX_i[:,1], YX_i[:,0]) * 180 / np.pi

           ids_16 =  (-45<= ang) & (ang <=45)
           ids_15 =  (45<= ang) & (ang <=135)
           ids_14_1 =  (135<= ang) & (ang <=180)
           ids_14_2 =  (-180<= ang) & (ang <=-135)
           ids_13 =  (-135<= ang) & (ang <=-45)

           aha_lv[z_idx[ids_13], y_i[ids_13], x_i[ids_13]] = 13
           aha_lv[z_idx[ids_14_1], y_i[ids_14_1], x_i[ids_14_1]] = 14
           aha_lv[z_idx[ids_14_2], y_i[ids_14_2], x_i[ids_14_2]] = 14
           aha_lv[z_idx[ids_15], y_i[ids_15], x_i[ids_15]] = 15
           aha_lv[z_idx[ids_16], y_i[ids_16], x_i[ids_16]] = 16

   elif ba_channel ==1:
      # --- Base
       for idy, y_i in enumerate(center_lv[:z_base,ba_channel]):
           idx = (y == y_i)
           z_i = z[idx]
           y_idx = y[idx]
           x_i = x[idx]
           zx_i = (np.vstack([z_i, x_i]).T - center_lv[idy,::2]).T
           # Cambiamos el sistema de coordenadas
           # NOTE: recordar que visualmente en el sistema de coord el origen esta arriba,
           # asi que invertimos los numeros de los segmentos
           # ids_6 <-> 5, 1 <-> 4, 3 <->2
           ZX_i = np.array(tmatrix * zx_i).T
           ang = np.arctan2(ZX_i[:,1], ZX_i[:,0]) * 180 / np.pi

           # Por una cuestion de redondeo voy a poner el = en todos ya que
           # pueden quedar agujeros sino se pone.
           ids_4 =  (60<= ang) & (ang <=120)
           ids_3 =  (120<= ang) & (ang <=180)
           ids_2 =  (-180<= ang) & (ang <=-120)
           ids_1 =  (-120<= ang) & (ang <=-60)
           ids_6 =  (-60<= ang) & (ang <=0)
           ids_5 =  (0<= ang) & (ang <=60)

           aha_lv[z_i[ids_1], y_idx[ids_1], x_i[ids_1]] = 1
           aha_lv[z_i[ids_2], y_idx[ids_2], x_i[ids_2]] = 2
           aha_lv[z_i[ids_3], y_idx[ids_3], x_i[ids_3]] = 3
           aha_lv[z_i[ids_4], y_idx[ids_4], x_i[ids_4]] = 4
           aha_lv[z_i[ids_5], y_idx[ids_5], x_i[ids_5]] = 5
           aha_lv[z_i[ids_6], y_idx[ids_6], x_i[ids_6]] = 6

       # --- Midd-cavity
       for idy, y_i in enumerate(center_lv[z_base:z_base+z_mid,ba_channel]):
           idx = (y == y_i)
           z_i = z[idx]
           y_idx = y[idx]
           x_i = x[idx]
           zx_i = (np.vstack([z_i, x_i]).T - center_lv[z_base+idy,::2]).T
           # Cambiamos el sistema de coordenadas
           # NOTE: recordar que visualmente en el sistema de coord el origen esta arriba,
           # asi que invertimos los numeros de los segmentos
           # ids_12 <-> 11, 7 <-> 10, 8 <->9
           ZX_i = np.array(tmatrix * zx_i).T
           ang = np.arctan2(ZX_i[:,1], ZX_i[:,0]) * 180 / np.pi

           ids_10 =  (60<= ang) & (ang <=120)
           ids_9 =  (120<= ang) & (ang <=180)
           ids_8 =  (-180<= ang) & (ang <=-120)
           ids_7 =  (-120<= ang) & (ang <=-60)
           ids_12 =  (-60<= ang) & (ang <=0)
           ids_11 =  (0<= ang) & (ang <=60)

           aha_lv[z_i[ids_7], y_idx[ids_7], x_i[ids_7]] = 7
           aha_lv[z_i[ids_8], y_idx[ids_8], x_i[ids_8]] = 8
           aha_lv[z_i[ids_9], y_idx[ids_9], x_i[ids_9]] = 9
           aha_lv[z_i[ids_10], y_idx[ids_10], x_i[ids_10]] = 10
           aha_lv[z_i[ids_11], y_idx[ids_11], x_i[ids_11]] = 11
           aha_lv[z_i[ids_12], y_idx[ids_12], x_i[ids_12]] = 12


       # --- Appex NOTE: revisar que siempre el apex es el mas chico en z
       for idy, y_i in enumerate(center_lv[z_base+z_mid:,ba_channel]):
           idx = (y == y_i)
           z_i = z[idx]
           y_idx = y[idx]
           x_i = x[idx]
           zx_i = (np.vstack([z_i, x_i]).T - center_lv[z_base+z_mid+idy,::2]).T
           # Cambiamos el sistema de coordenadas
           # NOTE: recordar que visualmente en el sistema de coord el origen esta arriba,
           # asi que invertimos los numeros de los segmentos
           # ids_13 <-> 15
           ZX_i = np.array(tmatrix * zx_i).T
           ang = np.arctan2(ZX_i[:,1], ZX_i[:,0]) * 180 / np.pi

           ids_16 =  (-45<= ang) & (ang <=45)
           ids_15 =  (45<= ang) & (ang <=135)
           ids_14_1 =  (135<= ang) & (ang <=180)
           ids_14_2 =  (-180<= ang) & (ang <=-135)
           ids_13 =  (-135<= ang) & (ang <=-45)

           aha_lv[z_i[ids_13], y_idx[ids_13], x_i[ids_13]] = 13
           aha_lv[z_i[ids_14_1], y_idx[ids_14_1], x_i[ids_14_1]] = 14
           aha_lv[z_i[ids_14_2], y_idx[ids_14_2], x_i[ids_14_2]] = 14
           aha_lv[z_i[ids_15], y_idx[ids_15], x_i[ids_15]] = 15
           aha_lv[z_i[ids_16], y_idx[ids_16], x_i[ids_16]] = 16



   elif ba_channel ==2:
       raise Exception('Unimplemented !!')


   aha_img = sitk.GetImageFromArray(aha_lv)
   aha_img.CopyInformation(myo)

   return aha_img


def add_ahaInfo2pd(aha_img, pd):
    aha_coord_pd = tvtk.PolyData()
    aha_coord_pd.deep_copy(pd)

    aha_lv = sitk.GetArrayFromImage(aha_img)
    z, y, x = np.where(aha_lv>0)

    aha = tvtk.UnsignedCharArray()
    aha.from_array(aha_lv[z,y,x].flatten())
    aha.number_of_components=1
    aha.name = 'aha'

    aha_coord_pd.point_data.add_array(aha)

    return aha_coord_pd



