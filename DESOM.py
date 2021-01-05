# This is largely a copy from Forest. I removed parts that weren't relevant to this particular
# problem and merged some functions from another file here for simplicity.

import os
import csv
from time import time
import numpy as np

import tensorflow as tf
from keras.models import Model
from keras import optimizers

from SOM import SOMLayer
from conv1DAE import cnn_1dae

def som_loss(weights, distances):
    """
    SOM loss

    # Arguments
        weights: weights for the weighted sum, Tensor with shape `(n_samples, n_prototypes)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, Tensor with shape `(n_samples, n_prototypes)`
    # Return
        SOM reconstruction loss
    """
    return tf.reduce_mean(tf.reduce_sum(weights*distances, axis=1))

def kmeans_loss(y_pred, distances):
    """
    k-means reconstruction loss

    # Arguments
        y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, numpy.array with shape `(n_samples, n_prototypes)`
    # Return
        k-means reconstruction loss
    """
    return np.mean([distances[i, y_pred[i]] for i in range(len(y_pred))])

def quantization_error(d):
    """
    Calculate k-means quantization error (internal DESOM function)
    """
    return d.min(axis=1).mean()

def topographic_error(d, map_size):
    """
    Calculate SOM topographic error (internal DESOM function)
    Topographic error is the ratio of data points for which the two best matching units are not neighbors on the map.
    """
    h, w = map_size

    def is_adjacent(k, l):
        return (abs(k//w-l//w) == 1 and abs(k % w - l % w) == 0) or (abs(k//w-l//w) == 0 and abs(k % w - l % w) == 1)
    btmus = np.argsort(d, axis=1)[:, :2]  # best two matching units
    return 1.-np.mean([is_adjacent(btmus[i, 0], btmus[i, 1]) for i in range(d.shape[0])])

class DESOM:
    """
    Deep Embedded Self-Organizing Map (DESOM) model
    """

    def __init__(self, input_dims, map_size, latent):
        self.input_dims = input_dims    # expects integer, number of data points in a sample
        self.map_size = map_size        # expects a tuple, height and width (usually going to be square)
        self.n_prototypes = map_size[0]*map_size[1]
        self.latent=latent              # width at Autoencoder bottleneck
    
    def initialize(self):
        """
        Create DESOM architecture.
        """
        # Make autoencoder
        self.autoencoder, self.encoder, self.decoder = cnn_1dae(input_dims=self.input_dims, latent_dims=self.latent)
        
        # Attach SOM layer
        som_layer = SOMLayer(self.map_size, name='SOM')(self.encoder.output)

        # Create DESOM model
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.autoencoder.output, som_layer])
   
    @property
    def prototypes(self):
        """
        Returns SOM code vectors
        """
        return self.model.get_layer(name='SOM').get_weights()[0]

    def compile(self, gamma, optimizer):
        """
        Compile DESOM model

        # Arguments
            gamma: coefficient of SOM loss
            optimizer: optimization algorithm
        """
        self.model.compile(loss={'decoder_0': 'mse', 'SOM': som_loss},
                           loss_weights=[1, gamma],
                           optimizer=optimizer)
    
    def load_weights(self, weights_path):
        """
        Load pre-trained weights of DESOM model

        # Arguments
            weight_path: path to weights file (.h5)
        """
        self.model.load_weights(weights_path)

    def init_som_weights(self, X):
        """
        Initialize weights of the SOM layer using the samples in X.

        # Arguments
            X: numpy array containing training set or batch
        """
        sample = X[np.random.choice(X.shape[0], size=self.n_prototypes, replace=False)]
        encoded_sample = self.encode(sample)
        self.model.get_layer(name='SOM').set_weights([encoded_sample])
        
    def encode(self, x):
        """
        Encoding function. Extract latent features from hidden layer

        # Arguments
            x: data point
        # Return
            encoded (latent) data point
        """
        return self.encoder.predict(x)
    
    def decode(self, x):
        """
        Decoding function. Decodes encoded features from latent space

        # Arguments
            x: encoded (latent) data point
        # Return
            decoded data point
        """
        return self.decoder.predict(x)
    
    def predict(self, x):
        """
        Predict best-matching unit using the output of SOM layer

        # Arguments
            x: data point
        # Return
            index of the best-matching unit
        """
        _, d = self.model.predict(x, verbose=0)
        return d.argmin(axis=1)
    
    def map_dist(self, y_pred):
        """
        Calculate pairwise Manhattan distances between cluster assignments and map prototypes (rectangular grid topology)
        
        # Arguments
            y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
        # Return
            pairwise distance matrix (map_dist[i,k] is the Manhattan distance on the map between assigned cell of data point i and cell k)
        """
        labels = np.arange(self.n_prototypes)
        tmp = np.expand_dims(y_pred, axis=1)
        d_row = np.abs(tmp-labels)//self.map_size[1]
        d_col = np.abs(tmp%self.map_size[1]-labels%self.map_size[1])
        return d_row + d_col
    
    def neighborhood_function(self, x, T):
        """
        SOM neighborhood function (gaussian neighborhood)

        # Arguments
            x: distance on the map
            T: temperature parameter
        # Return
            neighborhood weight
        """
        return np.exp(-(x**2)/(T**2))
    
    def fit(self, X_train,
            iterations=10000,
            som_iterations=10000,
            eval_interval=100,
            batch_size=256,
            Tmax=10,
            Tmin=0.1,
            decay='exponential',
            save_dir='results/tmp'):
        """
        Training procedure. # Model is saved only at end.

        # Arguments
           X_train: training set
           iterations: number of training iterations # (must be >= som_iterations)
           som_iterations: number of iterations where SOM neighborhood is decreased
           eval_interval: evaluate metrics on training/validation batch every eval_interval iterations
           batch_size: training batch size
           Tmax: initial temperature parameter
           Tmin: final temperature parameter
           decay: type of temperature decay ('exponential' or 'linear')
           save_dir: path to existing directory where weights and logs are saved
        """

        # Logging file
        logfile = open(save_dir + '/desom_log.csv', 'w')
        fieldnames = ['iter', 'T', 'L', 'Lr', 'Lsom', 'Lkm', 'Ltop', 'quantization_err', 'topographic_err', 'latent_quantization_err', 'latent_topographic_err']
        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()

        # Set and compute some initial values
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

            if ite % eval_interval == 0:
                # Initialize log dictionary
                logdict = dict(iter=ite, T=T)

                # Get SOM weights and decode to original space
                decoded_prototypes = self.decode(self.prototypes)

                # Evaluate losses and metrics
                logdict['L'] = loss[0]
                logdict['Lr'] = loss[1]
                logdict['Lsom'] = loss[2]
                logdict['Lkm'] = kmeans_loss(y_pred, d)
                logdict['Ltop'] = loss[2] - logdict['Lkm']
                logdict['latent_quantization_err'] = quantization_error(d)
                logdict['latent_topographic_err'] = topographic_error(d, self.map_size)
                d_original = np.square((np.expand_dims(X_batch, axis=1) - decoded_prototypes)).sum(axis=2)
                logdict['quantization_err'] = quantization_error(d_original)
                logdict['topographic_err'] = topographic_error(d_original, self.map_size)
                    
                logwriter.writerow(logdict)

        # Save the final model
        logfile.close()
        print('saving model to:', save_dir + '/DESOM_model_final.h5')
        self.model.save_weights(save_dir + '/DESOM_model_final.h5')
