"""
File: feature_tracking.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://github.com/curiale
Description: Funciones auxiliares para feature tracking
"""

#import os
import numpy as np
import SimpleITK as sitk
#from natsort import natsorted

#from config import *


# Callback used to show registration metrics
def command_iteration_demons(method) :
    print("{0:3} = {1:10.7f}".format(method.GetElapsedIterations(),
        method.GetMetric()))



def command_iteration_bspline(method) :
   print("{0:3} = {1:10.7f}".format(method.GetOptimizerIteration(),
     method.GetMetricValue()))
#    print("{0:3} = Metric: {1:10.5f} \tOp. Position: ".format(method.GetOptimizerIteration(),
#      method.GetMetricValue()), len(method.GetOptimizerPosition()))
#  #print("\t#: ", len(method.GetOptimizerPosition()))

def command_multi_iteration_bspline(method) :
 print("--------- Resolution Changing ---------")




def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """
    Args:
        image: The image we want to resample.
        shrink_factor: A number greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma: Sigma for Gaussian smoothing, this is in physical (image spacing) units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma and shrink factor.
    """
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz/float(shrink_factor[i]) + 0.5) for i, sz in enumerate(original_size)]
    new_spacing = [((original_sz-1)*original_spc)/(new_sz-1)
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(),
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0,
                         image.GetPixelID())



def multiscale_demons(registration_algorithm, niter,
                      fixed_image, moving_image, initial_transform = None,
                      shrink_factors=None, smoothing_sigmas=None):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors: Shrink factors relative to the original image's size.
        smoothing_sigmas: Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns:
        SimpleITK.DisplacementFieldTransform
    """
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        dim = fixed_image.GetDimension()
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            if type(shrink_factor) is not list:
                shrink_factor = np.repeat(shrink_factor, dim).astype(float)

            if type(smoothing_sigma) is not list:
                smoothing_sigma = np.repeat(smoothing_sigma, dim).astype(float)

            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform,
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(),
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])

    # Run the registration.
    registration_algorithm.SetNumberOfIterations(int(niter[0]))

    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1],
                                                                moving_images[-1],
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.
    idl=1
    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1]))):
        print("--------- Resolution Changing ---------")
        registration_algorithm.SetNumberOfIterations(int(niter[idl]))
        initial_displacement_field = sitk.Resample (initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
        idl +=1
    return sitk.DisplacementFieldTransform(initial_displacement_field)




def feature_tracking_method(F, M, method, F_mask=None, M_mask=None, niter=100,
        demons_std=[1,1,1], demons_update_std=None, shrink_factors=None,
        smoothing_sigmas=None, bs_physical_spacing=None,
        bs_gradientConvergenceTolerance=1e-5,
        bs_maximumNumberOfCorrections=5,
        bs_maximumNumberOfFunctionEvaluations=2000,
        bs_costFunctionConvergenceFactor=1e+7,
        bs_sampling_percentage=0.2):

    if method.lower() == 'demons':

        demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        #demons_filter = sitk.DiffeomorphicDemonsRegistrationFilter()
        #demons_filter.SetNumberOfIterations(niter)
        #demons_filter.SetIntensityDifferenceThreshold(2)
        # NOTE: Los dos parametros de smooth son fundamentales para calcular el
        # esfuerzo ya que la deformacion deberia ser bastante elastica. Revisar
        # bien estos parametros.
        # Gaussian smooth of the deformation field
        demons_filter.SetSmoothDisplacementField(True) # True by defautl
        demons_filter.SetStandardDeviations(demons_std) # Pixel coordinates. [1,1,1] by default
        # Gaussian smooth of the update
        if demons_update_std is not None:
            demons_filter.SetSmoothUpdateField(True) # False by defualt
            demons_filter.SetUpdateFieldStandardDeviations(demons_update_std) # Pixels coordinate.  [1,1,1] by defualt

        demons_filter.AddCommand( sitk.sitkIterationEvent,
                lambda: command_iteration_demons(demons_filter) )

        # Run the registration.
        if shrink_factors is not None:
            # Multi-resolution
            tx = multiscale_demons(registration_algorithm=demons_filter,
                            niter=niter,
                            fixed_image = F,
                            moving_image = M,
                            shrink_factors = shrink_factors,
                            smoothing_sigmas = smoothing_sigmas)
            dfield = tx.GetDisplacementField()
        else:
            demons_filter.SetNumberOfIterations(int(niter))
            dfield = demons_filter.Execute(F,M)

    elif method.lower() == 'bspline':
         # FFD - Bspline
        #Transformation
        # Determine the number of BSpline control points using the physical spacing we want for the control grid.
        image_physical_size = [size*spacing for size,spacing in zip(F.GetSize(), F.GetSpacing())]
        mesh_size = [int(image_size/grid_spacing + 0.5) \
                    for image_size,grid_spacing in zip(image_physical_size,bs_physical_spacing)]
        tx = sitk.BSplineTransformInitializer(F, mesh_size )

        R = sitk.ImageRegistrationMethod()
        # Metric
        R.SetMetricAsMeanSquares()
        #R.SetMetricAsANTSNeighborhoodCorrelation(4);
        #R.SetMetricAsMattesMutualInformation(50)
        #R.SetMetricAsCorrelation()

        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(bs_sampling_percentage) # Igual para todos los niveles
        #R.SetMetricSamplingPercentagePerLevel([0.01, 0.01])

        if F_mask is not None:
            R.SetMetricFixedMask(F_mask)
        if M_mask is not None:
            R.SetMetricMovingMask(M_mask)


        #Transformation
        R.SetInitialTransform(tx)
        R.SetInterpolator(sitk.sitkLinear)

        # NOTE: Es un tema que el Shrink factor sea igual para cada dimension porque
        # nuestra imagen no es isotropica. Ver de hacer lo que hicimos para el
        # demons para poder hacer multinivel.
        if shrink_factors is not None:
            R.SetShrinkFactorsPerLevel(shrink_factors)
            R.SetSmoothingSigmasPerLevel(smoothing_sigmas)
            R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


        # Optimization
        R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=bs_gradientConvergenceTolerance,
                maximumNumberOfCorrections=bs_maximumNumberOfCorrections,
                maximumNumberOfFunctionEvaluations=bs_maximumNumberOfFunctionEvaluations,
                costFunctionConvergenceFactor=bs_costFunctionConvergenceFactor,
                numberOfIterations=niter)

        #R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4,
        #       numberOfIterations=40, gradientMagnitudeTolerance=1e-8 )
        #R.SetOptimizerAsGradientDescentLineSearch(5.0, 100,
        #        convergenceMinimumValue=1e-4, convergenceWindowSize=5)

        # LBFGSB No usa escala
        #R.SetOptimizerScalesFromPhysicalShift( )
        #R.SetOptimizerScales(np.ones(np.prod(mesh_size))) # No logro quitar el warning


        R.AddCommand( sitk.sitkIterationEvent,
                lambda: command_iteration_bspline(R) )
        R.AddCommand( sitk.sitkMultiResolutionIterationEvent,
                lambda: command_multi_iteration_bspline(R) )
        tfm = R.Execute(F, M)

        toDisplacementFilter = sitk.TransformToDisplacementFieldFilter()
        toDisplacementFilter.SetReferenceImage(F)
        dfield = toDisplacementFilter.Execute(tfm)

    # NOTE: Voy a copiar el dfield en una nueva imagen porque por algun
    # motivo se esta corompiendo dfield al retornar de la funcion cuando uso
    # demons

    df = sitk.GetImageFromArray(sitk.GetArrayFromImage(dfield),
            isVector=True)
    df.CopyInformation(dfield)

    return df




