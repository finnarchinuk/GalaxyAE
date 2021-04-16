

import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, BatchNormalization, LeakyReLU
from keras.models import Model
from keras import optimizers
from keras import backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from DESOM2_architectures import DESOM, cnn_2dae, som_loss, kmeans_loss
from SOM import SOMLayer

parser = argparse.ArgumentParser()
parser.add_argument('--save_path',type=str)
parser.add_argument('--gamma',default=2e-4,type=float)
parser.add_argument('--map_size', nargs='+', default=[10, 10], type=int)
parser.add_argument('--iterations', default=30_000, type=int)
parser.add_argument('--som_iterations', default=25_000, type=int)
parser.add_argument('--batch_size', default = 128, type=int)
parser.add_argument('--Tmax',default=15,type=float)
parser.add_argument('--Tmin',default=0.1,type=float)
parser.add_argument('--lr',default=5e-4,type=float)
parser.add_argument('--seed',type=int)
args = parser.parse_args()

# -------------------------------- Log into Job Report ----------------------
print('map_size:',args.map_size)
print('gamma:',args.gamma)
print('lr:',args.lr)
print('iterations:',args.iterations)
print('som_iteration:',args.som_iterations)
print('batch_size:',args.batch_size)
print('T: '+str(args.Tmax)+' -> '+str(args.Tmin))
print('seed:',args.seed)

np.random.seed(args.seed)
tf.random.set_seed(args.seed)

#----------------------------------- Prepare Data (Fingerprints) ------------------------
X_data=np.load('Fingerprints/mar28_e.npy')# shape should be (4609,225)
X_data=MinMaxScaler().fit_transform(X_data.T).T
print('Applying MinMaxScaling to Fingerprints')

# Reshape
X_data=X_data.reshape(-1,15,15,1)

# Train using 90%
X_train,X_test=train_test_split(X_data,train_size=0.9)

# These will be used for plotting.
l_train_results=list()
l_test_results=list()

#----------------------------------- Test Multiple Latent Dimensions ---------------------------
LATENTS=[50,75,100,125,150,175,200]

for quick_latent in LATENTS:
    som = DESOM(input_dims= X_data.shape[1], map_size = args.map_size, latent_dims=quick_latent)
    optimizer = optimizers.Adam(args.lr)
    som.initialize()
    som.compile(args.gamma, optimizer)
    som.init_som_weights(X_train)
    som.fit(X_train, iterations = args.iterations, som_iterations = args.som_iterations,
            batch_size= args.batch_size, Tmax= args.Tmax, Tmin= args.Tmin, decay='exponential',
            save_path='results/som2_testing/'+args.save_path+'/')
    ae,r=som.model.predict(X_test)
    r=np.argmin(r,axis=1)
    
    
''' Things get dicey below (due to logging changes)'''

    # Save LR plots
    fig,ax=plt.subplots(2,1,sharex=True)
    ax[0].semilogy(Lossr)
    ax[1].semilogy(Losssom)
    ax[1].set_xlabel('epochs (more or less)')
    plt.savefig('results/som2_testing/{}/z{}_LR.png'.format(args.save_path,quick_latent))
    plt.clf()

    # Log AutoEncoder Reconstruction error for Test set and Training set
    print('MSE (test set) {}. latent: {}'.format(
        mean_squared_error(X_test.reshape(-1,225),ae.reshape(-1,225)),
        quick_latent))
    
    # Remove?: Won't work if logging properly.
    # print('MSE (train set):{}'.format(Lossr[-1]))
    
    # Reconstruction example
    plt.figure()
    plt.plot(X_test[0].reshape(225),label='input')
    plt.plot(ae[0].reshape(225),label='recon')
    plt.title('Model Reconstruction')
    plt.legend()
    plt.savefig('results/som2_testing/{}/z{}_reconstruction_example.png'.format(args.save_path,quick_latent)
    plt.clf()
    
    

    l_train_results.append(Lossr[-1])
    l_test_results.append(mean_squared_error(X_test.reshape(-1,225),ae.reshape(-1,225)))



plt.figure()
plt.plot(LATENTS,l_test_results,label='test set')
plt.plot(LATENTS,l_train_results,label='train set')
plt.xlabel('latent widths')
plt.ylabel('mse')
plt.legend()
plt.savefig('results/som2_testing/'+args.save_path+'/variable_latents.png')
plt.clf()

som.model.summary()
