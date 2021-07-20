'''
Module for training DESOM
'''

print('\n\n running DESOM_train')
#------------------Imports-----------------------
import argparse
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

from keras import optimizers
from DESOM import DESOM
from utils import normalize_4050_region

#----------------------Arguments----------------------
parser = argparse.ArgumentParser()

parser.add_argument('--map_size', nargs='+', default=[15, 15], type=int)
parser.add_argument('--gamma', default=5e-4, type=float, help='coefficient of self-organizing map loss')
parser.add_argument('--iterations', default=125000, type=int)
parser.add_argument('--som_iterations', default=100000, type=int)

parser.add_argument('--model_batch_size', default=128, type=int)
parser.add_argument('--Tmax', default=15.0, type=float)
parser.add_argument('--Tmin', default=0.7, type=float)
parser.add_argument('--save_path', type=str)

parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--latent', default = 128, type=int)
parser.add_argument('--seed', default = 0, type=int)

args = parser.parse_args()

# --------- Log training conditions -------
print('args')
print('save_path:', args.save_path)
print('map size:', args.map_size)
print('som epochs:', args.som_iterations)
print('both epochs:', args.iterations)
print('gamma:', args.gamma)
print('T:', args.Tmax, '->', args.Tmin)
print('lr:', args.lr)
print('latent:', args.latent)
print('seed: ', args.seed)
print('batch_size:', args.model_batch_size)
my_save_path = args.save_path

# --------- Path
SUFFIX = 'june11b'
x_train_path = 'x_train1m_' + SUFFIX + '.fits'
y_train_path = 'y_train1m_' + SUFFIX + '.fits'

x_test_path = 'x_test1m_' + SUFFIX + '.fits'
y_test_path = 'y_test1m_' + SUFFIX + '.fits'

print('using ' + x_train_path + ' for training')

#----------------------------Load DATA and specify ENCODER DIMENSIONS(ie.input dimensions)-------------------
x_train = Table.read(x_train_path)

input_dims = x_train['raw'][0].shape[0]

print('Normalizing X_train by 4050-4150 angstrom region')
normalize_4050_region(x_train, y_train, w_train)

print('Train size:',x_train['raw'].shape)
np.random.seed(args.seed)
np.random.shuffle(X_train) # an extra shuffle before training.

#-------------------------------------------- Architecture  and Fit -----------------------------------------------   
#Define, Initialize, and Compile DESOM
som = DESOM(input_dims = input_dims, # Data width: 4544
            map_size = args.map_size, # SOM size: (15,15)
            latent = args.latent) # Latent width of AE: 256
#Specify Gamma and Optimizer
gamma = args.gamma
optimizer = optimizers.Adam(args.lr)
#Initialize DESOM
som.initialize()
#Compile
som.compile(gamma, optimizer)


#Train DESOM
som.init_som_weights(x_train['raw'])
som.fit(x_train['raw'],
        iterations = args.iterations,
        som_iterations = args.som_iterations,
        batch_size= args.batch_size,
        Tmax= args.Tmax,
        Tmin= args.Tmin,
        save_dir='models/desom1/'+my_save_path+'/')

#------------------------------------------------- Plot Diagnostics --------------------------------------------------
map_size = som.map_size

# ---------------------- PLOT DECODED PROTOTYPES ----------------
decoded_prototypes = som.decode(som.prototypes)
fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(10,10))
for k in range(map_size[0] * map_size[1]):
    x = decoded_prototypes[k]
    ax[k // map_size[1]][k % map_size[1]].plot(x)
    ax[k // map_size[1]][k % map_size[1]].axis('off')
plt.subplots_adjust(hspace=1.05, wspace=1.05)
plt.savefig('results/desom1/'+my_save_path+'/prototypes.png')


#--------------------------- EVALUATE and PLOT ----------------------------------
x_test = Table.read(x_test_path) # LOAD TEST SET
y_test = Table.read(y_test_path)

print('Normalizing X_test by 4050-4150 angstrom region')
normalize_4050_region(x_test, y_test, w_test)


# ----------------- calculate bmu for test set ----------------
recons, tmp_bmus = som.model.predict(x_test['raw'])
tmp_bmus2 = np.argmin(tmp_bmus,axis=1)
y_test['bmu'] = tmp_bmus2


# ----------------------- ML_r, Heightmap, and Distribution -----------------------
# (ML_r is a measure of mass to light given a spectrum. It's a nice sanity check)

attr_av = np.zeros(map_size[0] * map_size[1])
heightmap = np.zeros(map_size[0] * map_size[1])

for bmu in range(map_size[0] * map_size[1]):
    masked = (y_test['bmu'] == int(bmu)) #remove int?
    temp_list = y_test['ML_r'][masked]
    attr_av[bmu] = np.median(np.log(temp_list)) #log for ML_r
    heightmap[bmu] = y_test[masked].sum()

plt.figure() 
plt.imshow(attr_av.reshape(map_size))
plt.title('log ML_r average')
plt.colorbar()
plt.savefig('results/desom1/'+my_save_path+'/log_ML_r_median.png')
plt.clf()
#
plt.figure()
plt.imshow(heightmap.reshape(map_size))
plt.title('x_test heightmap')
plt.colorbar()
plt.savefig('results/desom1/'+my_save_path+'/heightmap.png')
plt.clf()
#
plt.figure()
plt.xlabel('number of spectra in node')
plt.ylabel('number of nodes')
plt.hist(heightmap, bins = 20)
plt.title('x_test distribution')
plt.savefig('results/desom1/'+my_save_path+'/heightmap_distribution.png')


# --------------------- Systematic errors in reconstruction ? -------------------

print('test mse:',mean_squared_error(recons, x_test['raw']))

# calculate bias in reconstruction error
print('delta_mean:', np.mean(recons - x_test['raw']))
print('delta_median:', np.median(recons - x_test['raw']))
delta_mean = np.mean(recons-x_test['raw'],axis=1)
delta_median = np.median(recons-x_test['raw'],axis=1)

plt.clf()
fig,ax=plt.subplots(2,1)
ax[0].scatter(y_test['ML_r'], delta_mean, s=4, alpha=0.2)
ax[0].axhline(0, c='red', alpha=0.5)
ax[0].set_xlabel('ML_r')
ax[0].set_ylabel('bias')

ax[1].scatter(y_test['meansn2'], delta_mean, s=4, alpha=0.2)
ax[1].axhline(0, c='red', alpha=0.5)
ax[1].set_xlabel('noise')
ax[1].set_ylabel('bias')
plt.savefig('results/desom1/'+my_save_path+'/recon_bias.png')






# --------- calculate reconstruction bias and ML_r/heightmap for training set ------
print('starting plots using train sets\n')

# ------------- calculate reconstruction error --------
y_train = Table.read(y_train_path)

recons, tmp_bmus = som.model.predict(x_train['raw'])
tmp_bmus2 = np.argmin(tmp_bmus, axis=1)
y_train['bmu'] = tmp_bmus2

# -------- plot ML_r and heightmap --------
attr_av = np.zeros(map_size[0] * map_size[1])
heightmap = np.zeros(map_size[0] * map_size[1])

for bmu in range(map_size[0] * map_size[1]):
    masked = (y_train['bmu'] == int(bmu))
    temp_list = y_train['ML_r'][masked]
    attr_av[bmu] = np.median(np.log(temp_list))
    heightmap[bmu] = y_train[masked].sum()

plt.figure()
plt.imshow(attr_av.reshape(map_size))
plt.title('log ML_r average')
plt.colorbar()
plt.savefig('results/desom1/'+my_save_path+'/log_ML_r_median_training_set.png')
plt.clf()
#
plt.figure()
plt.imshow(heightmap.reshape(map_size))
plt.title('x_train heightmap')
plt.colorbar()
plt.savefig('results/desom1/'+my_save_path+'/heightmap_training_set.png')
plt.clf()


# ----------- recon bias --------------
print('train mse:', mean_squared_error(recons, x_train['raw']))

# calculate bias in reconstruction error
print('delta_mean (train):', np.mean(recons - x_train['raw']))
print('delta_median (train):', np.median(recons - x_train['raw']))
delta_mean = np.mean(recons - x_train['raw'], axis=1)
delta_median = np.median(recons - x_train['raw'], axis=1)

fig,ax=plt.subplots(2,1)
ax[0].scatter(y_train['ML_r'], delta_mean, s=4, alpha=0.2)
ax[0].axhline(0,c='red',alpha=0.1)
ax[0].set_xlabel('ML_r')
ax[0].set_ylabel('bias')

ax[1].scatter(y_train['meansn2'], delta_mean, s=4, alpha=0.2)
ax[1].axhline(0,c='red',alpha=0.1)
ax[1].set_xlabel('noise')
ax[1].set_ylabel('bias')
plt.savefig('results/desom1/'+my_save_path+'/recon_bias_training_set.png')
plt.clf()
