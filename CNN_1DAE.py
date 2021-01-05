''' defines the Autoencoder for DESOM-1. '''

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Reshape, Flatten
from keras.models import Model
from keras import backend as K

def CNN_1DAE(input_dims,latent_dims= 64):
    #------------------- Encoder--------------    
    input_layer = Input(shape=(input_dims,), name='input')
    reshaped_layer = Reshape((input_dims, 1))(input_layer) #Conv1D requires a channel axis

    x = Conv1D(16, 4, activation='relu', padding='same')(reshaped_layer) 
    x = MaxPooling1D(pool_size=4)(x) 
    x = Conv1D(16, 4, activation='relu', padding='same')(x) 
    x = MaxPooling1D(pool_size=4)(x) 

    #----------Store dimensions before we flatten.
    shape_before_flattening = K.int_shape(x)[1:]
    #Unpack
    dims, fmaps = shape_before_flattening
    num_neurons = dims*fmaps 

    #Wrap up encoder
    x = Flatten()(x)
    encoded = Dense(latent_dims, activation='relu', name='z')(x)  #Last layer in Encoder
    #Get the latent dimension
    z_dims =  K.int_shape(encoded)[1] #LATENT using K-backend

    #Define model
    encoder = Model(input_layer, encoded, name='encoder') #DEFINE ENCODER
              
    #------------------ Rest of Autoencoder--------------------

    x = Dense(num_neurons, activation='relu',name='decoder_7')(encoded)
    x = Reshape(shape_before_flattening, name='decoder_6')(x)    #Reshape for Convolutions
    x = UpSampling1D(4, name = 'decoder_5')(x)
    x = Conv1D(16, 4, activation='relu', padding='same', name = 'decoder_4')(x) 
    x = UpSampling1D(4, name = 'decoder_3')(x) 
    x = Conv1D(16, 4, activation='relu', padding='same', name = 'decoder_2')(x) 
    x = Conv1D(1, 4, activation='linear', padding='same', name='decoder_1')(x) 
    decoded = Flatten(name='decoder_0')(x)

    autoencoder = Model(input_layer, decoded, name='autoencoder') #DEFINE AUTOENCODER

    #----------------------- Standalone Decoder------------------
    # Create input for decoder model
    encoded_input = Input(shape=(latent_dims,))

    #Unpack layers
    decoded = autoencoder.get_layer('decoder_7')(encoded_input)
    decoded = autoencoder.get_layer('decoder_6')(decoded)
    decoded = autoencoder.get_layer('decoder_5')(decoded)
    decoded = autoencoder.get_layer('decoder_4')(decoded)
    decoded = autoencoder.get_layer('decoder_3')(decoded)
    decoded = autoencoder.get_layer('decoder_2')(decoded)
    decoded = autoencoder.get_layer('decoder_1')(decoded)
    decoded = autoencoder.get_layer('decoder_0')(decoded)

    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder') #DEFINE DECODER

    return (autoencoder, encoder, decoder)
