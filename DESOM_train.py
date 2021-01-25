'''
Module for training DESOM
'''

#------------------Imports-----------------------
import argparse
import os
import numpy as np
import time
import sys

from DESOM import DESOM
import matplotlib.pyplot as plt
from keras import optimizers
from astropy.table import Table

#----------------------Arguments----------------------
parser = argparse.ArgumentParser()

parser.add_argument('--map_size', nargs='+', default=[15, 15], type=int)
parser.add_argument('--gamma', default=1e-4, type=float, help='coefficient of self-organizing map loss')
parser.add_argument('--iterations', default=125000, type=int)
parser.add_argument('--som_iterations', default=10000, type=int)
parser.add_argument('--model_batch_size', default=16, type=int)
parser.add_argument('--Tmax', default=15.0, type=float)
parser.add_argument('--Tmin', default=2.0, type=float)
parser.add_argument('--save_path',type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--latent', default=256, type=int)
parser.add_argument('--seed', default=0, type=int)

#Obtain args
args = parser.parse_args()

# --------- Log training conditions -------
print('args')
print('save_path:',args.save_path)
print('map size:',args.map_size)
print('som epochs:',args.som_iterations) #only adjusting SOM weights
print('both epochs:',args.iterations) #both refers to adjustments of both the SOM and AE components
print('gamma:',args.gamma)
print('T:',args.Tmax,'->',args.Tmin)
print('lr:',args.lr)
print('latent:',args.latent)
print('seed: ',args.seed)
my_save_path=args.save_path

#------------------------------------------------ Path names----------------
X_train_path='X_train400k.npy'
X_test_path ='X_test400k.npy'
Y_test_path ='Y_test400k.fits'

#---------------Load DATA and specify ENCODER DIMENSIONS(ie.input dimensions)
X_train=np.load(X_train_path) # These have been Standard Scaled.
input_dims=X_train.shape[1]

print('Train size:',X_train.shape)
print('Train size reduced to:',X_train.shape)
np.random.seed(args.seed)
np.random.shuffle(X_train) # an extra shuffle before training.

#-------------------------------------------- Architecture  and Fit -----------------------------------------------   
#Define, Initialize and Compile DESOM
som = DESOM(input_dims = input_dims, #spectrum with (4544)
            map_size = args.map_size, #SOM size (expects tuple)
            latent = args.latent)
som.initialize()
gamma = args.gamma
optimizer = optimizers.Adam(args.lr)
som.compile(gamma, optimizer)

#Train DESOM
som.init_som_weights(X_train)
som.fit(X_train,
        iterations = args.iterations,
        som_iterations = args.som_iterations,
        save_epochs = args.save_epochs,
        batch_size= args.model_batch_size,
        Tmax= args.Tmax,
        Tmin= args.Tmin,
        decay='exponential',
        save_dir='results/tmp/'+my_save_path+'/')

#------------------------------------------------- Plot Diagnostics --------------------------------------------------

map_size=som.map_size

# LOAD TEST SET #and look at subset
X=np.load(X_test_path)
Xmin=X[:20000]
Y_test=Table.read(Y_test_path)
Ymin=Y_test[:20000]

def get_distance_map(desom, X):
    return desom.map_dist(desom.predict(X))

#DISTRIBUTION INTO NODES
# Bin each spectrum into the best node
Ymin['bmu']=som.predict(Xmin)

# Mask each node
attr_av=np.zeros(map_size[0]*map_size[1])
heightmap=np.zeros(map_size[0]*map_size[1])
ATTR='ML_r'

for bmu in range(map_size[0]*map_size[1]):
    masked=(Ymin['bmu']==int(bmu))
    temp_list=Ymin[ATTR][masked]
    attr_av[bmu]=np.mean(np.log(temp_list)) #log for ML_r
    heightmap[bmu]=len(Ymin[masked])

plt.figure(0)
plt.imshow(attr_av.reshape(map_size))
plt.title('log ML_r average')
plt.colorbar()
plt.savefig('results/tmp/'+my_save_path+'/log_ML_r_average.png')
plt.clf()
#
plt.figure(1)
plt.imshow(heightmap.reshape(map_size))
plt.title('20k input heightmap')
plt.colorbar()
plt.savefig('results/tmp/'+my_save_path+'/heightmap.png')
plt.clf()
#
plt.figure(2)
plt.xlabel('number of spectra in node')
plt.ylabel('number of nodes')
plt.hist(heightmap,bins=20)
plt.title('heightmap 20k node distribution')
plt.savefig('results/tmp/'+my_save_path+'/heightmap_distribution.png')

