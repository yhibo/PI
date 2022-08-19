import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

def remove_extra_inputs_outputs(model, input_label, output_label):
   # Trim the model to get the unet trained
   # NOTE: the layer name is hardcoded according to the ACNN model and Doble
   # Unet
   #  layer = get_tensor(model, 'unet_output')
   linput = model.get_layer(input_label).output
   loutput = model.get_layer(output_label).output
   new_model = Model(linput, loutput)
   return new_model
