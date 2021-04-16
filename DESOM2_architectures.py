
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, LeakyReLU
from keras.models import Model
from keras import backend as K
from SOM import SOMLayer
import tensorflow as tf
import numpy as np

# contains three functions: 
# cnn_1dae(input_dims,latent_dims)
# both take (batch,15,15,1) as input (a 15x15 SOM with one channel)

# And DESOM (the som2 version)


def cnn_2dae(input_dims,latent_dims):
    '''
    Inputs: input_dims expects to be (15,15,1)
            latent_dims expects an integer: for example 200
    Returns: (Autoencoder, Encoder, Decoder)
    '''
    #------------------- Encoder--------------
    #Input layer
    input_layer = Input(shape=(15,15,1), name='input')
    x = Conv2D(32,(3,3),activation='relu',padding='same')(input_layer)
    x = Conv2D(4,(3,3),activation='relu',padding='same')(x)

    shape_before_flattening = K.int_shape(x)[1:]
    dims,fmaps,other=shape_before_flattening
    num_neurons=dims*fmaps*other
    x = Flatten()(x)
    x = Dense(latent_dims,name='z')(x)

    encoder = Model(input_layer, x, name='encoder') ########## ENCODER

    decoded = Dense(num_neurons,activation='linear',name='decoder_4')(x)
    decoded = Reshape((shape_before_flattening),name='decoder_3')(decoded)
    decoded = Conv2D(4,(3,3),activation='relu',padding='same',name='decoder_2')(decoded)
    decoded = Conv2D(32,(3,3),activation='relu',padding='same',name='decoder_1')(decoded)
    decoded = Conv2D(1,(1,1),activation='linear',name='decoder_0')(decoded)

    autoencoder = Model(input_layer, decoded, name='autoencoder') ######## AE
    
    #stand alone decoder
    encoded_input = Input(shape=(latent_dims,))
    decoded = autoencoder.get_layer('decoder_4')(encoded_input)
    decoded = autoencoder.get_layer('decoder_3')(decoded)
    decoded = autoencoder.get_layer('decoder_2')(decoded)
    decoded = autoencoder.get_layer('decoder_1')(decoded)
    decoded = autoencoder.get_layer('decoder_0')(decoded)
    
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder') ###### DECODER
    return (autoencoder, encoder, decoder)


def som_loss(weights, distances):
    return tf.reduce_mean(tf.reduce_sum(weights*distances, axis=1))

def kmeans_loss(y_pred, distances):
    return np.mean([distances[i, y_pred[i]] for i in range(len(y_pred))])

class DESOM:
    def __init__(self, input_dims, map_size, latent_dims):
        self.input_dims = input_dims
        self.map_size = map_size
        self.n_prototypes = map_size[0]*map_size[1]
        self.latent_dims=latent_dims

    def initialize(self):
        self.autoencoder, self.encoder, self.decoder = cnn_2dae(input_dims=self.input_dims,latent_dims=self.latent_dims)
        som_layer = SOMLayer(self.map_size, name='SOM')(self.encoder.output)
        # Create DESOM model
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.autoencoder.output, som_layer])
    @property
    def prototypes(self):
        return self.model.get_layer(name='SOM').get_weights()[0]

    def compile(self, gamma, optimizer):
        self.model.compile(loss={'decoder_0': 'mse', 'SOM': som_loss},
                           loss_weights=[1, gamma],
                           optimizer=optimizer)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def init_som_weights(self, X):
        sample = X[np.random.choice(X.shape[0], size=self.n_prototypes, replace=False)]
        encoded_sample = self.encode(sample)
        self.model.get_layer(name='SOM').set_weights([encoded_sample])

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, x):
        return self.decoder.predict(x)

    def predict(self, x):
        _, d = self.model.predict(x, verbose=0)
        return d.argmin(axis=1)

    def map_dist(self, y_pred):
        labels = np.arange(self.n_prototypes)
        tmp = np.expand_dims(y_pred, axis=1)
        d_row = np.abs(tmp-labels)//self.map_size[1]
        d_col = np.abs(tmp%self.map_size[1]-labels%self.map_size[1])
        return d_row + d_col

    def neighborhood_function(self, x, T):
        return np.exp(-(x**2)/(T**2))

    def fit(self, X_train,
            iterations=60000, som_iterations=55000,
            eval_interval=100, save_epochs=5, batch_size=128,
            Tmax=15, Tmin=0.1, decay='exponential', save_path='default_path'):

        logfile = open(save_dir + '/desom2_log_{}.csv'.format(self.latent_dims), 'w')
        fieldnames = ['iter', 'T', 'L', 'Lr', 'Lsom', 'Lkm', 'Ltop']
        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()
      
        index = 0
        for ite in range(iterations):
            # Get training and validation batches
            if (index + 1) * batch_size >= X_train.shape[0]:
                X_batch = X_train[index * batch_size::]
                index = 0
            else:
                X_batch = X_train[index * batch_size:(index + 1) * batch_size]
                index += 1

            # Compute cluster assignments for batches
            _, d = self.model.predict(X_batch)
            y_pred = d.argmin(axis=1)

            # Update temperature parameter
            if ite < som_iterations:
                if decay == 'exponential':
                    T = Tmax*(Tmin/Tmax)**(ite/(som_iterations-1))
                elif decay == 'linear':
                    T = Tmax - (Tmax-Tmin)*(ite/(som_iterations-1))

            # Compute topographic weights batches
            w_batch = self.neighborhood_function(self.map_dist(y_pred), T)

            # Train on batch
            loss = self.model.train_on_batch(X_batch, [X_batch, w_batch])

            if ite % 50 == 0:
                # Initialize log dictionary
                logdict = dict(iter=ite, T=T)
                
                # Get SOM weights and decode to original space
                decoded_prototypes = self.decode(self.prototypes)

''' TO DO: Test the fix below. '''
                # Evaluate losses and metrics
                logdict['L'] = loss[0]
                logdict['Lr'] = loss[1]
                logdict['Lsom'] = loss[2]
                logdict['Lkm'] = kmeans_loss(y_pred, d)
                logdict['Ltop'] = loss[2] - logdict['Lkm']
                logwriter.writerow(logdict)
        logfile.close()
        self.model.save_weights(save_path + 'DESOM_model_final_{}.h5'.format(self.latent_dims))
