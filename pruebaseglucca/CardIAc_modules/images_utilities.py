# ----------------------------------------------------
# Ing. Lucca Dellazoppa - Instituto Balseiro 
# lucca.dellazoppa.m@gmail.com
# ---------------------------------------------------- 
import numpy as np
import vtk
import SimpleITK as sitk
import vtk.util.numpy_support as numpy_support

def normalize_image(I):
    # Scale intensity to [0, 255]
    img_p = I.astype('float32')
    img_p -= img_p.min(axis=(0,1))[np.newaxis, np.newaxis]
    img_p /= img_p.max(axis=(0,1))[np.newaxis, np.newaxis]
    img_p *= 255
    img_p -= img_p.mean(axis=(0,1))[np.newaxis, np.newaxis]
    img_p /= img_p.std(axis=(0,1))[np.newaxis, np.newaxis] + 1e-7
    return img_p
        
def NpArray2VTK(npVol, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), scalarType=True):
    # Numpy coordiante system convention [row, col, j] = [y x z]
    # VTK coordinate system convention [z y x]
    # Nota: npVol.shape -> [row, col, k] = [y, x, z]
    if len(npVol.shape) == 2:
        npVol = npVol.transpose((1, 0))
    elif len(npVol.shape) == 3 and not scalarType:
        # Dejamos la dim. de datos donde esta
        npVol = npVol.transpose((1, 0, 2))
    else:
        transp_value = (2, 0, 1)
        for i in range(3, len(npVol.shape)):
            transp_value += (i, )
        npVol = npVol.transpose(transp_value)
    # npVol Ahora esta en [z, y, x]
    # Partimos el if en dos por claridad
    # Nota: los datos tiene que estar en z, x, y pero el shape hay que pasarlo
    # como x, y, z. Ahora invertimos el shape para que este en x, y, z

    if len(npVol.shape) == 2:
        shape = list(npVol.shape + (1,))
        origin = (0.0, origin[0], origin[1])
        spacing = (1.0, spacing[0], spacing[1])
    else:
        shape = list(npVol.shape)
    shape.reverse()
    # Convertimos el array numpy to VTK para eso pasamos los datos en 1D
    npVol = npVol.ravel()
    vtk_data = numpy_support.numpy_to_vtk(npVol)
    # Creamos una vtkImageData
    img = vtk.vtkImageData()
    img.SetDimensions(shape[0], shape[1], shape[2])
    img.SetSpacing(spacing[0], spacing[1], spacing[2])
    img.SetOrigin(origin[0], origin[1], origin[2]),
    img.GetPointData().SetScalars(vtk_data)
    img.GetPointData().GetArray(0).SetName('ImageData')
    # Posiblemente se haga una copia con el transpose de npVol y luego
    # fuera de la funcion se puede borrar,  de esta forma se pierde el
    # contenido de vtk_data (si el collector de memoria decide borrar el npVol)
    # Por ese motivo TENGO que devolver el npVol y mantenerlo vivo fuera de la
    # Funcion
    return (img, npVol)

def sitk2vtk(self,img):
    I = sitk.GetArrayFromImage(img)
    vtk_data = numpy_support.numpy_to_vtk(I.ravel())
    # Creamos una vtkImageData
    vtkimg = vtk.vtkImageData()
    vtkimg.SetDimensions(img.GetSize())
    vtkimg.SetSpacing(img.GetSpacing())
    vtkimg.SetOrigin(img.GetOrigin())
    vtkimg.GetPointData().SetScalars(vtk_data)
    vtkimg.GetPointData().GetArray(0).SetName('ImageData')
    return vtkimg