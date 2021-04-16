''' defines the Autoencoder for DESOM-1. '''

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Reshape, Flatten, LeakyReLU
from keras.models import Model
from keras import backend as K

ALPHA=0.1
def cnn_1dae(input_dims,latent_dims= 64):
    #------------------- Encoder--------------
    input_layer = Input(shape=(input_dims,), name='input')
    reshaped_layer = Reshape((input_dims, 1))(input_layer) #Conv1D requires a channel axis

    x = Conv1D(64, 6,strides=1, padding='same')(reshaped_layer) 
    x = LeakyReLU(alpha=ALPHA)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(64, 6,strides=1, padding='same')(x) 
    x = LeakyReLU(alpha=ALPHA)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, 6, strides=1, padding='same')(x) 
    x = LeakyReLU(alpha=ALPHA)(x)
    x = MaxPooling1D(pool_size=2)(x) 
    
    x = Conv1D(64, 6, strides=1,padding='same')(x) 
    x = LeakyReLU(alpha=ALPHA)(x)
    x = MaxPooling1D(pool_size=2)(x) 

    #----------Store dimensions before we flatten.
    shape_before_flattening = K.int_shape(x)[1:]
    #Unpack
    dims, fmaps = shape_before_flattening
    num_neurons = dims*fmaps 

    #Wrap up encoder
    x = Flatten()(x)
    x = Dense(latent_dims,)(x)  #Last layer in Encoder
    encoded = LeakyReLU(alpha=ALPHA,name='z')(x)
    #Get the latent dimension
    z_dims =  K.int_shape(encoded)[1] #LATENT using K-backend

    #Define model
    encoder = Model(input_layer, encoded, name='encoder') #DEFINE ENCODER
              
    #------------------ Rest of Autoencoder--------------------

    x = Dense(num_neurons,name='decoder_11')(encoded)
    x = LeakyReLU(alpha=ALPHA,name='decoder_11a')(x)
    x = Reshape(shape_before_flattening, name='decoder_10')(x)    #Reshape for Convolutions

    x = UpSampling1D(2, name = 'decoder_9')(x)
    x = Conv1D(64, 6, padding='same', name = 'decoder_8')(x) 
    x = LeakyReLU(alpha=ALPHA,name='decoder_8a')(x)
    
    x = UpSampling1D(2, name = 'decoder_7')(x) 
    x = Conv1D(64, 6, padding='same', name = 'decoder_6')(x) 
    x = LeakyReLU(alpha=ALPHA,name='decoder_6a')(x)
    
    x = UpSampling1D(2, name = 'decoder_5')(x) 
    x = Conv1D(64, 6, padding='same', name = 'decoder_4')(x) 
    x = LeakyReLU(alpha=ALPHA,name='decoder_4a')(x)
    
    x = UpSampling1D(2, name = 'decoder_3')(x) 
    x = Conv1D(64, 6, padding='same', name = 'decoder_2')(x)
    x = LeakyReLU(alpha=ALPHA,name='decoder_2a')(x)
    
    x = Conv1D(1, 1, activation='linear', padding='same', name='decoder_1')(x)
    decoded = Flatten(name='decoder_0')(x)

    autoencoder = Model(input_layer, decoded, name='autoencoder') #DEFINE AUTOENCODER

    #----------------------- Standalone Decoder------------------
    # Create input for decoder model
    encoded_input = Input(shape=(latent_dims,))

    #Unpack layers
    decoded = autoencoder.get_layer('decoder_11')(encoded_input)
    decoded = autoencoder.get_layer('decoder_11a')(decoded)
    decoded = autoencoder.get_layer('decoder_10')(decoded)
    decoded = autoencoder.get_layer('decoder_9')(decoded)
    decoded = autoencoder.get_layer('decoder_8')(decoded)
    decoded = autoencoder.get_layer('decoder_8a')(decoded)
    decoded = autoencoder.get_layer('decoder_7')(decoded)
    decoded = autoencoder.get_layer('decoder_6')(decoded)
    decoded = autoencoder.get_layer('decoder_6a')(decoded)
    decoded = autoencoder.get_layer('decoder_5')(decoded)
    decoded = autoencoder.get_layer('decoder_4')(decoded)
    decoded = autoencoder.get_layer('decoder_4a')(decoded)
    decoded = autoencoder.get_layer('decoder_3')(decoded)
    decoded = autoencoder.get_layer('decoder_2')(decoded)
    decoded = autoencoder.get_layer('decoder_2a')(decoded)
    decoded = autoencoder.get_layer('decoder_1')(decoded)
    decoded = autoencoder.get_layer('decoder_0')(decoded)

    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder') #DEFINE DECODER
    print(autoencoder.summary())
    print('AE - LeakyReLU Alpha: {}'.format(ALPHA))
    return (autoencoder, encoder, decoder)
