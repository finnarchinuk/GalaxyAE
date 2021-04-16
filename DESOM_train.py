'''
Module for training DESOM
'''

#------------------Imports-----------------------
import argparse
import numpy as np
import time
import sys

from DESOM import DESOM
import matplotlib.pyplot as plt
from keras import optimizers
from astropy.table import Table
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#----------------------Arguments----------------------
parser = argparse.ArgumentParser()

parser.add_argument('--map_size', nargs='+', default=[15, 15], type=int)
parser.add_argument('--gamma', default=1e-4, type=float, help='coefficient of self-organizing map loss')
parser.add_argument('--iterations', default=125000, type=int)
parser.add_argument('--som_iterations', default=100000, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--Tmax', default=15.0, type=float)
parser.add_argument('--Tmin', default=2.0, type=float)
parser.add_argument('--save_path',type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--latent', default=256, type=int)
parser.add_argument('--seed', default=1, type=int)

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

#------------------------------------------------ Path names----------------------------------
X_train_path='X_train_1m.npy'
Y_train_path='Y_train_1m.npy'
X_test_path ='X_test_1m.npy'
Y_test_path ='Y_test_1m.fits'
print('Using '+ X_train_path+ ' for training.')

#----------------------------Load DATA and specify ENCODER DIMENSIONS(ie.input dimensions)-------------------
X_train=np.load(X_train_path) # These have been Standard Scaled.
input_dims=X_train.shape[1]

print('Apply a MinMaxScaler to training set')
X_train=MinMaxScaler().fit_transform(X_train.T).T

print('Train size:',X_train.shape)
np.random.seed(args.seed)
np.random.shuffle(X_train) # an extra shuffle before training.

#-------------------------------------------- Architecture  and Fit -----------------------------------------------   
#Define, Initialize, and Compile DESOM
som = DESOM(input_dims = input_dims, # Data width: 4544
            map_size = args.map_size, # SOM size: (15,15)
            latent = args.latent) # Latent width of AE: 256
som.initialize()
gamma = args.gamma
optimizer = optimizers.Adam(args.lr)
som.compile(gamma, optimizer)

#Train DESOM
som.init_som_weights(X_train)
som.fit(X_train,
        iterations = args.iterations,
        som_iterations = args.som_iterations,
        eval_interval=100,
        batch_size= args.batch_size,
        Tmax= args.Tmax,
        Tmin= args.Tmin,
        decay='exponential',
        save_dir='results/tmp/'+args.save_path+'/')

#------------------------------------------------- Plot Diagnostics --------------------------------------------------
map_size=som.map_size
#--------------------------- Test Set ---------------------
# Load Test set
Xmin=np.load(X_test_path) # These have been Standard Scaled.
print('Applying a MinMaxScaler to Test set.')
Xmin=MinMaxScaler().fit_transform(Xmin.T).T
Ymin=Table.read(Y_test_path)

# Run spectra through the model and capture outputs.
recons,tmp_bmus=som.model.predict(Xmin)

# Bin each spectrum into the best node on the SOM.
Ymin['bmu']=np.argmin(tmp_bmus,axis=1)

#------------------------- Calculate heightmap and log ML_r at each node --------------------
attr_av=np.zeros(map_size[0]*map_size[1])
heightmap=np.zeros(map_size[0]*map_size[1])
ATTR='ML_r'

for bmu in range(map_size[0]*map_size[1]):
    masked=(Ymin['bmu']==int(bmu))
    temp_list=Ymin[ATTR][masked]
    attr_av[bmu]=np.mean(np.log(temp_list)) # log for ML_r
    heightmap[bmu]=len(Ymin[masked])

#------------------------ Plot Heightmap and log ML_r at each node --------------------
plt.figure()
plt.imshow(attr_av.reshape(map_size))
plt.title('log ML_r average')
plt.colorbar()
plt.savefig('results/tmp/'+args.save_path+'/log_ML_r_average.png')
plt.clf()
#
plt.figure()
plt.imshow(heightmap.reshape(map_size))
plt.title('100k input heightmap')
plt.colorbar()
plt.savefig('results/tmp/'+args.save_path+'/heightmap.png')
plt.clf()

#--------------------------- Plot distribution of samples in each node ------------------
plt.figure()
plt.xlabel('number of spectra in node')
plt.ylabel('number of nodes')
plt.hist(heightmap,bins=20)
plt.title('heightmap 100k node distribution')
plt.savefig('results/tmp/'+args.save_path+'/heightmap_distribution.png')
plt.clf()

#----------------------------- Log reconstruction error for Test Set ---------------------
print('MSE Test set:',mean_squared_error(recons,Xmin))

#------------------------ Calculate bias in reconstruction error ------------------
print('delta_mean:',np.mean(recons-Xmin))
print('delta_median:',np.median(recons-Xmin))
delta_mean=np.mean(recons-Xmin,axis=1)
delta_median=np.median(recons-Xmin,axis=1)

#-------------------------- Plot bias in reconstruction error --------------------
fig,ax=plt.subplots(2,1)
ax[0].scatter(Ymin['ML_r'],delta_mean,s=4,alpha=0.2)
ax[0].axhline(0,c='red',alpha=0.5)

ax[1].scatter(Ymin['meansn2'],delta_mean,s=4,alpha=0.2)
ax[1].axhline(0,c='red',alpha=0.5)
plt.savefig('results/tmp/'+my_save_path+'/recon_bias.png')
plt.clf()

#-------------------------- Training set (for comparison) -------------------------
# Load
Y_train=Table.read(Y_train_path)

# Run spectra through network and capture outputs
recons,tmp_bmus=som.model.predict(X_train)
Y_train['bmu']=np.argmin(tmp_bmus,axis=1)


#------------------------ Calculate Heightmap and log ML_r at each node --------------------
attr_av=np.zeros(map_size[0]*map_size[1])
heightmap=np.zeros(map_size[0]*map_size[1])
ATTR='ML_r'

for bmu in range(map_size[0]*map_size[1]):
    masked=(Y_train['bmu']==int(bmu))
    temp_list=Y_train[ATTR][masked]
    attr_av[bmu]=np.mean(np.log(temp_list)) #log for ML_r
    heightmap[bmu]=len(Y_train[masked])

#------------------------ Plot Heightmap and log ML_r at each node --------------------
plt.figure()
plt.imshow(attr_av.reshape(map_size))
plt.title('log ML_r average')
plt.colorbar()
plt.savefig('results/tmp/'+my_save_path+'/log_ML_r_average_training_set.png')
plt.clf()
#
plt.figure()
plt.imshow(heightmap.reshape(map_size))
plt.title('900k input heightmap')
plt.colorbar()
plt.savefig('results/tmp/'+my_save_path+'/heightmap_training_set.png')
plt.clf()

#----------------------------- Log reconstruction error for Test Set ---------------------
print('MSE Train set:',mean_squared_error(recons,X_train))

#------------------------ Calculate bias in reconstruction error ------------------
print('delta_mean (train):',np.mean(recons-X_train))
print('delta_median (train):',np.median(recons-X_train))
delta_mean=np.mean(recons-X_train,axis=1)
delta_median=np.median(recons-X_train,axis=1)

#-------------------------- Plot bias in reconstruction error --------------------
fig,ax=plt.subplots(2,1)
ax[0].scatter(Y_train['ML_r'],delta_mean,s=4,alpha=0.2)
ax[0].axhline(0,c='red',alpha=0.1)

ax[1].scatter(Y_train['meansn2'],delta_mean,s=4,alpha=0.2)
ax[1].axhline(0,c='red',alpha=0.1)
plt.savefig('results/tmp/'+args.save_path+'/recon_bias_training_set.png')
plt.clf()

#----------------------------- Decoded Prototypes as grid ----------------------------
decoded_prototypes = som.decode(som.prototypes)
fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(40,40))
for k in range(map_size[0] * map_size[1]):
    x = decoded_prototypes[k]
    ax[k // map_size[1]][k % map_size[1]].plot(x)
    ax[k // map_size[1]][k % map_size[1]].axis('off')
plt.subplots_adjust(hspace=1.05, wspace=1.05)
plt.savefig('results/tmp/'+args.save_path+'/prototypes.png')
plt.clf()
