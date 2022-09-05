"""
File: strain_from_motion.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://github.com/curiale
Description: Compute the Lagrangian, eulerian and infinitesimal strain from the
displacement field. Also, it is computed the strain rate.
"""


import numpy as np
from scipy.ndimage import interpolation


def temporal_strain_correction(ie_aha):

    ie_aha_fix = []
    for i in range(len(ie_aha)):
        ie = ie_aha[i]
        nframes = ie.shape[-1] - 1
        ien = ie[:, -1]

        m = ien/nframes
        ie_f = ie - m[:, np.newaxis] * np.arange(nframes+1)

        ie_aha_fix.append(ie_f)
    return ie_aha_fix

def cine_dense_strain(df, Icoord, mask, ba_channel=0):
    tracking_3d = len(df) == 3
    if tracking_3d:
        return cine_dense_strain3D(df, Icoord, mask, ba_channel)
    else:
        return cine_dense_strain2D(df, Icoord, mask, ba_channel)


def cine_dense_strain2D(df, Icoord, mask, ba_channel):
    ## Compute the local strain for each point in the mask and move the points
    ## according to the deformation field

    # En teoria el df esta en mm
    u = df[0] # Esta en [z, y, x]
    v = df[1]

    u_z, u_y, u_x = np.gradient(u) # Devuelve en orden row, col, ...
    v_z, v_y, v_x = np.gradient(v)


    # Ahora calculamos el esfuerzo infinitesimal
    e_xx = u_x
    e_yy = v_y
    e_xy = .5*(u_y + v_x)
    e_yx = e_xy


    iec = np.zeros(mask.shape, dtype=float)
    ier = np.zeros(mask.shape, dtype=float)
    ierc = np.zeros(mask.shape, dtype=float)

    ec = np.zeros(mask.shape, dtype=float)
    er = np.zeros(mask.shape, dtype=float)
    erc = np.zeros(mask.shape, dtype=float)

    Ec = np.zeros(mask.shape, dtype=float)
    Er = np.zeros(mask.shape, dtype=float)
    Erc = np.zeros(mask.shape, dtype=float)


    # Calculamos el esfuerzo punto a punto solo dentro de la mascara
    rr, cc, jj = np.where(mask>0)

    for r, c, j in zip(rr,cc, jj):
        # NOTE: Local Coord. estan en (x,y,z) si ba_channel = 0.
        # Si es ba_channel = 1 --> [x,z,y]
        # Si es ba_channel = 2 --> [z,y,x]
        if ba_channel == 0:
            c_c = Icoord[1][r,c,j,:] # x,y,z
            c_r = Icoord[2][r,c,j,:] # x,y,z
        elif ba_channel == 1:
            c_c = Icoord[1][r,c,j,:]
            c_r = Icoord[2][r,c,j,:]
            # pasamos a (x,y,z)
            c_c = np.asarray([c_c[0], c_c[2], c_c[1]])
            c_r = np.asarray([c_r[0], c_r[2], c_r[1]])
        elif ba_channel == 2:
            c_c = Icoord[1][r,c,j,:]
            c_r = Icoord[2][r,c,j,:]
            # pasamos a (x,y,z)
            c_c = np.asarray([c_c[2], c_c[1], c_c[0]])
            c_r = np.asarray([c_r[2], c_r[1], c_r[0]])

        # 1) Infinitesimal strain
        # Hago la proyeccion sobre c_c y c_r
        inf_e = np.array([[e_xx[r,c,j], e_xy[r,c,j]],
                            [e_yx[r,c,j], e_yy[r,c,j]]
                            ])
        iec[r,c,j] = np.dot(np.dot(c_c[:2], inf_e), c_c[:2]) # 2D
        ier[r,c,j] = np.dot(np.dot(c_r[:2], inf_e), c_r[:2]) # 2D
        ierc[r,c,j] = np.dot(np.dot(c_r[:2], inf_e), c_c[:2]) # 2D

        # 2) Eulerian strain
        gu = np.array([[u_x[r,c,j], u_y[r,c,j]],
                        [v_x[r,c,j], v_y[r,c,j]]
                        ])
        e = 0.5 * (gu + gu.T - np.dot(gu,gu.T))

        # Hago la proyeccion sobre c_c y c_r
        ec[r,c,j] = np.dot(np.dot(c_c[:2], e), c_c[:2]) # 2D
        er[r,c,j] = np.dot(np.dot(c_r[:2], e), c_r[:2]) # 2D
        erc[r,c,j] = np.dot(np.dot(c_r[:2], e), c_c[:2]) # 2D


        # 3) Lagrangian strain
        E = 0.5 * (gu + gu.T + np.dot(gu.T, gu))

        # Hago la proyeccion sobre c_c y c_r
        Ec[r,c,j] = np.dot(np.dot(c_c[:2], E), c_c[:2]) # 2D
        Er[r,c,j] = np.dot(np.dot(c_r[:2], E), c_r[:2]) # 2D
        Erc[r,c,j] = np.dot(np.dot(c_r[:2], E), c_c[:2]) # 2D


    return iec, ier, ierc, ec, er, erc, Ec, Er, Erc

def cine_dense_strain3D(df, Icoord, mask, ba_channel):

    ## Compute the local strain for each point in the mask and move the points
    ## according to the deformation field

    # df[2] = df[2] * 8/1.25
    # print(df.shape)
    # df = interpolation.zoom(df, [1, 8/1.25, 1, 1], order=1)
    # mask = interpolation.zoom(mask, [8/1.25, 1, 1], order=0)
    # Icoord = interpolation.zoom(Icoord, [1, 8/1.25, 1, 1, 1], order=1)
    # print(df.shape)

    # En teoria el df esta en mm
    u = df[0] # Esta en [z, y, x]
    v = df[1]
    w = df[2]
    
    u_z, u_y, u_x = np.gradient(u) # Devuelve en orden row, col, ...
    v_z, v_y, v_x = np.gradient(v)
    w_z, w_y, w_x = np.gradient(w)



    # Ahora calculamos el esfuerzo infinitesimal
    e_xx = u_x
    e_yy = v_y
    e_xy = .5*(u_y + v_x)
    e_yx = e_xy
    e_zz = w_z
    e_xz = .5*(u_z + w_x)
    e_zx = e_xz
    e_yz = .5*(v_z + w_y)
    e_zy = e_yz

    iec = np.zeros(mask.shape, dtype=float)
    ier = np.zeros(mask.shape, dtype=float)
    ierc = np.zeros(mask.shape, dtype=float)

    ec = np.zeros(mask.shape, dtype=float)
    er = np.zeros(mask.shape, dtype=float)
    erc = np.zeros(mask.shape, dtype=float)

    Ec = np.zeros(mask.shape, dtype=float)
    Er = np.zeros(mask.shape, dtype=float)
    Erc = np.zeros(mask.shape, dtype=float)


    # Calculamos el esfuerzo punto a punto solo dentro de la mascara
    rr, cc, jj = np.where(mask>0)

    iel = np.zeros(mask.shape, dtype=float)
    el = np.zeros(mask.shape, dtype=float)
    El = np.zeros(mask.shape, dtype=float)

    for r, c, j in zip(rr,cc, jj):
        # NOTE: Local Coord. estan en (x,y,z) si ba_channel = 0.
        # Si es ba_channel = 1 --> [x,z,y]
        # Si es ba_channel = 2 --> [z,y,x]
        if ba_channel == 0:
            c_l = Icoord[0][r,c,j,:] # x,y,z
            c_c = Icoord[1][r,c,j,:] # x,y,z
            c_r = Icoord[2][r,c,j,:] # x,y,z
        elif ba_channel == 1:
            c_l = Icoord[0][r,c,j,:] # x,y,z
            c_c = Icoord[1][r,c,j,:]
            c_r = Icoord[2][r,c,j,:]
            # pasamos a (x,y,z)
            c_l = np.asarray([c_l[0], c_l[2], c_l[1]])
            c_c = np.asarray([c_c[0], c_c[2], c_c[1]])
            c_r = np.asarray([c_r[0], c_r[2], c_r[1]])
        elif ba_channel == 2:
            c_l = Icoord[0][r,c,j,:] # x,y,z
            c_c = Icoord[1][r,c,j,:]
            c_r = Icoord[2][r,c,j,:]
            # pasamos a (x,y,z)
            c_l = np.asarray([c_l[2], c_l[1], c_l[0]])
            c_c = np.asarray([c_c[2], c_c[1], c_c[0]])
            c_r = np.asarray([c_r[2], c_r[1], c_r[0]])


        # 1) Infinitesimal strain
        # Hago la proyeccion sobre c_c y c_r
        inf_e = np.array([[e_xx[r,c,j], e_xy[r,c,j], e_xz[r,c,j]],
                            [e_yx[r,c,j], e_yy[r,c,j], e_yz[r,c,j]],
                            [e_zx[r,c,j], e_zy[r,c,j], e_zz[r,c,j]]
                            ])
        iec[r,c,j] = np.dot(np.dot(c_c, inf_e), c_c)
        ier[r,c,j] = np.dot(np.dot(c_r, inf_e), c_r)
        iel[r,c,j] = np.dot(np.dot(c_l, inf_e), c_l)
        ierc[r,c,j] = np.dot(np.dot(c_r, inf_e), c_c)

        # 2) Eulerian strain
        gu = np.array([[u_x[r,c,j], u_y[r,c,j], u_z[r,c,j]],
                        [v_x[r,c,j], v_y[r,c,j], v_z[r,c,j]],
                        [w_x[r,c,j], w_y[r,c,j], w_z[r,c,j]]
                        ])
        e = 0.5 * (gu + gu.T - np.dot(gu,gu.T))

        # Hago la proyecciones
        ec[r,c,j] = np.dot(np.dot(c_c, e), c_c)
        er[r,c,j] = np.dot(np.dot(c_r, e), c_r)
        el[r,c,j] = np.dot(np.dot(c_l, e), c_l)
        erc[r,c,j] = np.dot(np.dot(c_r, e), c_c)


        # 3) Lagrangian strain
        E = 0.5 * (gu + gu.T + np.dot(gu.T, gu))

        # Hago la proyeccion sobre c_c y c_r
        Ec[r,c,j] = np.dot(np.dot(c_c, E), c_c)
        Er[r,c,j] = np.dot(np.dot(c_r, E), c_r)
        El[r,c,j] = np.dot(np.dot(c_l, E), c_l)
        Erc[r,c,j] = np.dot(np.dot(c_r, E), c_c)

    # iec = interpolation.zoom(iec, [1.25/8, 1, 1], order=1)
    # ier = interpolation.zoom(ier, [1.25/8, 1, 1], order=1)
    # iel = interpolation.zoom(iel, [1.25/8, 1, 1], order=1)
    # ierc = interpolation.zoom(ierc, [1.25/8, 1, 1], order=1)
    # ec = interpolation.zoom(ec, [1.25/8, 1, 1], order=1)
    # er = interpolation.zoom(er, [1.25/8, 1, 1], order=1)
    # el = interpolation.zoom(el, [1.25/8, 1, 1], order=1)
    # erc = interpolation.zoom(erc, [1.25/8, 1, 1], order=1)
    # Ec = interpolation.zoom(Ec, [1.25/8, 1, 1], order=1)
    # Er = interpolation.zoom(Er, [1.25/8, 1, 1], order=1)
    # El = interpolation.zoom(El, [1.25/8, 1, 1], order=1)
    # Erc = interpolation.zoom(Erc, [1.25/8, 1, 1], order=1)
    
    return iec, ier, ierc, ec, er, erc, Ec, Er, Erc, iel, el, El





def cine_strains2mesh(mesh, aha_coord):
    ''' Calculamos el esfuerzo que sufre la malla. Para eso se espera que mesh
    tenga un campo con el desplazamiento instantaneo.
    '''
    from tvtk.api import tvtk

    c_c = aha_coord.point_data.get_array('circ')
    c_l = aha_coord.point_data.get_array('long')
    c_r = aha_coord.point_data.get_array('rad')

    # the output gradient tuple will be
    # {du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz} for an
    # input array {u, v, w}
    gf = tvtk.GradientFilter()
    gf.set_input_data(mesh)
    gf.set_input_scalars(0, 'disp') # 0:
    gf.update()

    mesh_g = gf.get_output()
    grad = mesh_g.point_data.get_array('Gradients').to_array()

    u_x = grad[:,0]
    u_y = grad[:,1]
    u_z = grad[:,2]

    v_x = grad[:,3]
    v_y = grad[:,4]
    v_z = grad[:,5]

    w_x = grad[:,6]
    w_y = grad[:,7]
    w_z = grad[:,8]

    # Ahora calculamos el esfuerzo infinitesimal
    e_xx = u_x
    e_yy = v_y
    e_zz = w_z

    e_xy = .5*(u_y + v_x)
    e_xz = .5*(u_z + w_x)
    e_yx = e_xy
    e_zx = e_xz

    e_yz = .5*(v_z + w_y)
    e_zy = e_yz

    npts = mesh.number_of_points

    iec = np.zeros(npts, dtype=float)
    ier = np.zeros(npts, dtype=float)
    iel = np.zeros(npts, dtype=float)
    ierc = np.zeros(npts, dtype=float)

    ec = np.zeros(npts, dtype=float)
    er = np.zeros(npts, dtype=float)
    el = np.zeros(npts, dtype=float)
    erc = np.zeros(npts, dtype=float)

    Ec = np.zeros(npts, dtype=float)
    Er = np.zeros(npts, dtype=float)
    El = np.zeros(npts, dtype=float)
    Erc = np.zeros(npts, dtype=float)


    for idp in range(npts):

        # 1) Infinitesimal strain
        # Hago la proyeccion sobre c_c, c_r y c_l
        inf_e = np.array([[e_xx[idp], e_xy[idp], e_xz[idp]],
                          [e_yx[idp], e_yy[idp], e_yz[idp]],
                          [e_zx[idp], e_zy[idp], e_zz[idp]]]
                          )
        iec[idp] = np.dot(np.dot(c_c[idp], inf_e), c_c[idp])
        ier[idp] = np.dot(np.dot(c_r[idp], inf_e), c_r[idp])
        iel[idp] = np.dot(np.dot(c_l[idp], inf_e), c_l[idp])
        ierc[idp] = np.dot(np.dot(c_r[idp], inf_e), c_c[idp])

        # 2) Eulerian strain
        gu = np.array([[u_x[idp], u_y[idp], u_z[idp]],
                       [v_x[idp], v_y[idp], v_z[idp]],
                       [w_x[idp], w_y[idp], w_z[idp]]]
                       )
        e = 0.5 * (gu + gu.T - np.dot(gu,gu.T))

        # Hago la proyeccion sobre c_c, c_r y c_l
        ec[idp] = np.dot(np.dot(c_c[idp], e), c_c[idp])
        er[idp] = np.dot(np.dot(c_r[idp], e), c_r[idp])
        el[idp] = np.dot(np.dot(c_l[idp], e), c_l[idp])
        erc[idp] = np.dot(np.dot(c_r[idp], e), c_c[idp])



        # 3) Lagrangian strain
        E = 0.5 * (gu + gu.T + np.dot(gu.T, gu))

        # Hago la proyeccion sobre c_c, c_r y c_l
        Ec[idp] = np.dot(np.dot(c_c[idp], E), c_c[idp])
        Er[idp] = np.dot(np.dot(c_r[idp], E), c_r[idp])
        El[idp] = np.dot(np.dot(c_l[idp], E), c_l[idp])
        Erc[idp] = np.dot(np.dot(c_r[idp], E), c_c[idp])


    # Agregamos el inf. strain
    earray = tvtk.FloatArray()
    earray.from_array(iec)
    earray.name = 'iec'

    mesh.point_data.add_array(earray)

    earray = tvtk.FloatArray()
    earray.from_array(ier)
    earray.name = 'ier'

    mesh.point_data.add_array(earray)

    earray = tvtk.FloatArray()
    earray.from_array(iel)
    earray.name = 'iel'

    mesh.point_data.add_array(earray)

    earray = tvtk.FloatArray()
    earray.from_array(ierc)
    earray.name = 'ierc'

    mesh.point_data.add_array(earray)

    # Eulerian
    earray = tvtk.FloatArray()
    earray.from_array(ec)
    earray.name = 'ec'

    mesh.point_data.add_array(earray)

    earray = tvtk.FloatArray()
    earray.from_array(er)
    earray.name = 'er'

    mesh.point_data.add_array(earray)

    earray = tvtk.FloatArray()
    earray.from_array(el)
    earray.name = 'el'

    mesh.point_data.add_array(earray)

    earray = tvtk.FloatArray()
    earray.from_array(erc)
    earray.name = 'erc'

    mesh.point_data.add_array(earray)


    # Lagrangian
    earray = tvtk.FloatArray()
    earray.from_array(Ec)
    earray.name = 'Ec'

    mesh.point_data.add_array(earray)

    earray = tvtk.FloatArray()
    earray.from_array(Er)
    earray.name = 'Er'

    mesh.point_data.add_array(earray)

    earray = tvtk.FloatArray()
    earray.from_array(El)
    earray.name = 'El'

    mesh.point_data.add_array(earray)

    earray = tvtk.FloatArray()
    earray.from_array(Erc)
    earray.name = 'Erc'

    mesh.point_data.add_array(earray)


    return mesh



def tag_inf_eulerian_lagrangian_strain(u, coord_i, mask):
    ''' Only circ. and radial strain are computed due to the tagging image used
    are 2D in the short axis'''

    # NOTE: Se calcula el esfuerzo infinitecimal a partir de los desplazamiento u
    # Ojo que siempre estamos trabajando en coord. de pixels. Luego
    # ver si no conviene normalizar a mm para poder comparar entre distintos
    # pacientes

    ux = u[0]
    uy = u[1]

    # Calculamos grad U
    u_xy, u_xx =  np.gradient(ux)  # Devuelve en orden g_row, g_col
    u_yy, u_yx =  np.gradient(uy)

    # Ahora calculamos el esfuerzo infinitesimal
    e_xx = u_xx
    e_yy = u_yy

    e_xy = .5*(u_xy + u_yx)
    e_yx = e_xy # e_yx = u_xy + u_yx


    iec = np.zeros(mask.shape, dtype=float)
    ier = np.zeros(mask.shape, dtype=float)
    ierc = np.zeros(mask.shape, dtype=float)

    ec = np.zeros(mask.shape, dtype=float)
    er = np.zeros(mask.shape, dtype=float)
    erc = np.zeros(mask.shape, dtype=float)

    Ec = np.zeros(mask.shape, dtype=float)
    Er = np.zeros(mask.shape, dtype=float)
    Erc = np.zeros(mask.shape, dtype=float)

    # Calculamos el esfuerzo punto a punto solo dentro de la mascara
    rr, cc = np.where(mask>0)
    for r, c in zip(rr,cc):
        # NOTE: Local Coord. estan en (x,y,z)
        # coord solo es 2D no hay esfuerzo long
        c_c = coord_i[1][r,c,:] # x,y,z
        c_r = coord_i[2][r,c,:] # x,y,z

        # 1) Infinitesimal strain
        # Hago la proyeccion sobre c_c y c_r
        inf_e = np.array([[e_xx[r,c], e_xy[r,c]],
                          [e_yx[r,c], e_yy[r,c]]])
        iec[r,c] = np.dot(np.dot(c_c[:2], inf_e), c_c[:2]) # 2D
        ier[r,c] = np.dot(np.dot(c_r[:2], inf_e), c_r[:2]) # 2D
        ierc[r,c] = np.dot(np.dot(c_r[:2], inf_e), c_c[:2]) # 2D

        # 2) Eulerian strain
        gu = np.array([[u_xx[r,c], u_xy[r,c]],
                       [u_yx[r,c], u_yy[r,c]]])
        e = 0.5 * (gu + gu.T - np.dot(gu,gu.T))

        # Hago la proyeccion sobre c_c y c_r
        ec[r,c] = np.dot(np.dot(c_c[:2], e), c_c[:2]) # 2D
        er[r,c] = np.dot(np.dot(c_r[:2], e), c_r[:2]) # 2D
        erc[r,c] = np.dot(np.dot(c_r[:2], e), c_c[:2]) # 2D


        # 3) Lagrangian strain
        E = 0.5 * (gu + gu.T + np.dot(gu.T, gu))

        # Hago la proyeccion sobre c_c y c_r
        Ec[r,c] = np.dot(np.dot(c_c[:2], E), c_c[:2]) # 2D
        Er[r,c] = np.dot(np.dot(c_r[:2], E), c_r[:2]) # 2D
        Erc[r,c] = np.dot(np.dot(c_r[:2], E), c_c[:2]) # 2D


    return iec, ier, ierc, ec, er, erc, Ec, Er, Erc




    ## Ahora calculamos el esfuerzo infinitesimal
    #df_ux_x,df_ux_y,df_ux_z = np.gradient(df[...,0])
    #df_uy_x,df_uy_y,df_uy_z = np.gradient(df[...,1])
    #df_uz_x,df_uz_y,df_uz_z = np.gradient(df[...,2])

    ## Por simplicidad lo vamos a calcular como eij = 0.5(uij+uji)
    #st = np.zeros(df_ux_x.shape + (3,3), dtype=float)
    #st[...,0,0] = df_ux_x
    #st[...,0,1] = .5*(df_ux_y + df_uy_x)
    #st[...,1,0] = st[...,0,1]
    #st[...,0,2] = .5*(df_ux_z + df_uz_x)
    #st[...,2,0] = st[...,0,2]
    #st[...,1,1] = df_uy_y
    #st[...,1,2] = .5*(df_uy_z + df_uz_y)
    #st[...,2,1] = st[...,1,2]
    #st[...,2,2] = df_uz_z

    ## Compute the local strain for each point in the mask and move the points
    ## according to the deformation field





    #ux = u[0]
    #uy = u[1]
    #uz = u[2]

    ## Calculamos grad U
    #u_xy, u_xx =  np.gradient(ux)  # Devuelve en orden g_row, g_col
    #u_yy, u_yx =  np.gradient(uy)

    #e_xx = u_xx
    #e_yy = u_yy

    #e_xy = .5*(u_xy + u_yx)
    #e_yx = e_xy # e_yx = u_xy + u_yx


    #iec = np.zeros(mask.shape, dtype=float)
    #ier = np.zeros(mask.shape, dtype=float)
    #ierc = np.zeros(mask.shape, dtype=float)

    #ec = np.zeros(mask.shape, dtype=float)
    #er = np.zeros(mask.shape, dtype=float)
    #erc = np.zeros(mask.shape, dtype=float)

    #Ec = np.zeros(mask.shape, dtype=float)
    #Er = np.zeros(mask.shape, dtype=float)
    #Erc = np.zeros(mask.shape, dtype=float)

    ## Calculamos el esfuerzo punto a punto solo dentro de la mascara
    #rr, cc = np.where(mask>0)
    #for r, c in zip(rr,cc):
    #    # NOTE: Local Coord. estan en (x,y,z)
    #    # coord solo es 2D no hay esfuerzo long
    #    c_c = coord_i[1][r,c,:] # x,y,z
    #    c_r = coord_i[2][r,c,:] # x,y,z

    #    # 1) Infinitesimal strain
    #    # Hago la proyeccion sobre c_c y c_r
    #    inf_e = np.array([[e_xx[r,c], e_xy[r,c]],
    #                      [e_yx[r,c], e_yy[r,c]]])
    #    iec[r,c] = np.dot(np.dot(c_c[:2], inf_e), c_c[:2]) # 2D
    #    ier[r,c] = np.dot(np.dot(c_r[:2], inf_e), c_r[:2]) # 2D
    #    ierc[r,c] = np.dot(np.dot(c_r[:2], inf_e), c_c[:2]) # 2D

    #    # 2) Eulerian strain
    #    gu = np.array([[u_xx[r,c], u_xy[r,c]],
    #                   [u_yx[r,c], u_yy[r,c]]])
    #    e = 0.5 * (gu + gu.T - np.dot(gu,gu.T))

    #    # Hago la proyeccion sobre c_c y c_r
    #    ec[r,c] = np.dot(np.dot(c_c[:2], e), c_c[:2]) # 2D
    #    er[r,c] = np.dot(np.dot(c_r[:2], e), c_r[:2]) # 2D
    #    erc[r,c] = np.dot(np.dot(c_r[:2], e), c_c[:2]) # 2D


    #    # 3) Lagrangian strain
    #    E = 0.5 * (gu + gu.T + np.dot(gu.T, gu))

    #    # Hago la proyeccion sobre c_c y c_r
    #    Ec[r,c] = np.dot(np.dot(c_c[:2], E), c_c[:2]) # 2D
    #    Er[r,c] = np.dot(np.dot(c_r[:2], E), c_r[:2]) # 2D
    #    Erc[r,c] = np.dot(np.dot(c_r[:2], E), c_c[:2]) # 2D


    #return iec, ier, ierc, ec, er, erc, Ec, Er, Erc



def strain_rate(u, coord_i, mask):
    ''' Only circ. and radial rate strain are computed due to the tagging image
    used are 2D in the short axis'''

    #TODO: sacar del paper de osman et al. sobre visualizacion

    ec = np.zeros(mask.shape, dtype=float)
    er = np.zeros(mask.shape, dtype=float)

    return ec, er


