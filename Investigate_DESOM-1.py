
######### CELL1
# LOAD DESOM MODEL AND X_TEST
import os
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from astropy.io import fits
from astropy.table import Table

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from DESOM import DESOM

X=np.load('X_test250k.npy')

#-------------------------------------------------- Paths -------------------------------------------------
root = os.getcwd()
full_save_path='results/tmp/oct13_x/'       #MODEL PATH
saved_weights = os.path.join(root, full_save_path)
model_path = os.path.join(saved_weights, 'DESOM_model_final.h5')

#------------------------------------------ Load the DESOM weights -------------------------------

def load_desom_trained(map_size, input_dims, model_path, latent):
    som = DESOM(input_dims= input_dims, map_size = map_size, latent=latent)
    som.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    som.load_weights(model_path)
    return som

som = load_desom_trained((15,15), X.shape[1], model_path, latent=256)
map_size = som.map_size

########### CELL2
''' Maps spectra to SOM, add this information to info table ''' 

#---------LOAD / SELECT SPECTRA (just looking at first 20k samples)----
Xmin=X[:20000,:]
Y_test=Table.read('Y_test250k.fits')
Ymin=Y_test[:20000]

# ---------- Bin each spectrum into the best node -------------
Ymin['bmu']=som.predict(Xmin)

########### CELL3
''' Colour SOM map by a parameter
Create a map showing distribution of spectra into node ("heightmap")'''
attr_av=np.zeros(map_size[0]*map_size[1])
heightmap=np.zeros(map_size[0]*map_size[1])

ATTR='ML_r' #This is used as a simple control

for bmu in range(map_size[0]*map_size[1]):
    masked=(Ymin['bmu']==int(bmu))
    temp_list=Ymin[ATTR][masked]
    attr_av[bmu]=np.nanmean(temp_list)
    heightmap[bmu]=len(Ymin[masked])

plt.figure(0)
plt.imshow(attr_av.reshape(map_size))
plt.title(ATTR+' average')
plt.colorbar()

plt.figure(2)
plt.imshow(heightmap.reshape(map_size))
plt.title('heightmap')
plt.colorbar()

############# CELL4
''' This cell adds some other useful variables to the available parameters.'''

'''The suffix of the mangaID'''
r=np.array(Ymin['mangaID'],dtype='str')
r2=np.zeros(len(r),dtype='int')
for i in range(len(r)):
    r2[i]=int(r[i].split('-')[1])
Ymin['suffix']=r2

''' Adds sSFR and logMS_dens_kpc to info'''
from astropy.cosmology import FlatLambdaCDM
cosmod = FlatLambdaCDM(name='Concordance',H0=70.0,Om0=0.3,Tcmb0=2.725, Neff=3.04, Ob0=0.0463)
#
pixscale = 0.000138889*3600. # arcsec (seems to be same for all cubes)
pixscale = pixscale/3600.*np.pi/180. # radians
comd_bax = cosmod.comoving_distance(Ymin['redshift']).value*1.0e3 #comoving dist in kpc
spaxsize_bax = pixscale*comd_bax/(1.0 + Ymin['redshift']) # kpc
spaxsize_bax = spaxsize_bax**2. # kpc^2
#
logMs_dens_kpc = Ymin['logMs_dens'] - np.log10(spaxsize_bax)  # solar mass per kpc^2
#
Ymin['logMS_dens_kpc']=logMs_dens_kpc
Ymin['sSFR']=Ymin['logSFR_dens']-Ymin['logMS_dens_kpc']

'''Simplify ML_r'''
Ymin['logML_r']=np.log(Ymin['ML_r'])


############ CELL5
''' distribution of parameters in SOM2.
Taking median to limit affect of outliers'''

ATTR=['BPTdist_Ka03','logML_r','ebv_stars','sSFR','r_re_light']

for m in range(len(ATTR)):
    plt.figure(m)
    temp_param2=np.zeros(som.map_size[0]*som.map_size[1])
    for i in range(som.map_size[0]*som.map_size[1]):
        temp_list=list()
        mask_bmu=(Ymin['bmu']==i)
        temp_list_of_gals=np.unique(Ymin['suffix'][mask_bmu])
        for k in range(len(temp_list_of_gals)):
            mask_gal=(Ymin['suffix']==temp_list_of_gals[k])
            temp_list.append(np.nanmedian(Ymin[ATTR[m]][mask_gal]))
        temp_param2[i]=np.nanmedian(temp_list)
    plt.imshow(temp_param2.reshape(som.map_size))
    plt.title(ATTR[m])
    plt.axis('off')
    plt.colorbar()
    
    
################ CELL6
''' residual reconstruction error '''
recons=som.autoencoder.predict(Xmin)
delta=recons-Xmin

''' Spread out by a couple variables'''
ATTR='logML_r'
fig,ax=plt.subplots(2,1,figsize=(10,8))
ax[0].scatter(Ymin[ATTR],np.average(delta,axis=1),s=4,alpha=0.4)
ax[0].set_xlabel(ATTR)
ax[0].set_ylabel('delta reconstuction error')

ATTR='meansn2'
ax[1].scatter(Ymin[ATTR],np.average(delta,axis=1),s=4,alpha=0.4)
ax[1].set_xlabel(ATTR)
ax[1].set_ylabel('delta reconstuction error')

''' bmu reshaped'''
temp_error=np.zeros(225)
for i in range(225):
    mask=(Ymin['bmu']==i)
    temp=delta[mask].mean(axis=1)
    temp_error[i]=temp.std()

plt.figure(2)
plt.imshow(temp_error.reshape(15,15))
plt.colorbar()
plt.title('AE reconstruction error')


#################### CELL7
''' Distribution of parameter in node'''

# ------------------------ PUT ALL VALUES INTO A LIST --------------
value=list()
ATTR='logML_r'

for bmu in range(map_size[0]*map_size[1]):
    mask=(Ymin['bmu']==int(bmu))
    value.append(Ymin[mask][ATTR])

# ---------------- LOOK AT DISTRIBUTION WITHIN A SPECIFIC NODE. ---------
node_index=14

print(len(value[node_index]),'spectra in node',node_index)
print('average:',np.nanmean(value[node_index]))
mask=(Ymin['bmu']==int(node_index))

plt.figure(0,figsize=(10,2))
plt.title(ATTR+' distribution')
plt.hist(Ymin[mask][ATTR],bins=50)
plt.xlim(np.nanmin(Ymin[mask][ATTR]),np.nanmax(Ymin[mask][ATTR]))

################ CELL8
'''Plot the decoded prototype for each node'''

decoded_prototypes = som.decode(som.prototypes)
fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(16,16))
for k in range(map_size[0] * map_size[1]):
    x = decoded_prototypes[k]
    ax[k // map_size[1]][k % map_size[1]].plot(x)
    ax[k // map_size[1]][k % map_size[1]].axis('off')
plt.subplots_adjust(hspace=1.05, wspace=1.05)


################# CELL9
''' Dotwise colouring to capture different values within a prototype '''
import matplotlib as mpl
import matplotlib.cm as cm

data_path='data/gal_split_100/'
MaNGA_SUFFIX=44219      #galaxy 1-44219

ATTR='r_re_light' #Color spectra by this attribute
#ATTR='logMS_dens_kpc'

# ------------------------ LOAD DATA --------------------
test_gal=np.load(data_path+str(MaNGA_SUFFIX)+'.npy')    #load spectra for this galaxy
test_gal=StandardScaler().fit_transform(test_gal.T).T   #normalize spectra
#
Y_info=Table.read(data_path+str(MaNGA_SUFFIX)+'_info.fits') #load data

#----------------- FIX OR ADD DATA -------------------------
''' add BMU '''
Y_info['bmu']=som.predict(test_gal)

'''The suffix of the mangaID. (used for filing)'''
r=np.array(Y_info['mangaID'],dtype='str')
r2=np.zeros(len(r),dtype='int')
for i in range(len(r)):
    r2[i]=int(r[i].split('-')[1])
Y_info['suffix']=r2    

''' Simplify ML_r'''
Y_info['logML_r']=np.log(Y_info['ML_r'])

''' fix MS_dens, add sSFR'''
from astropy.cosmology import FlatLambdaCDM
cosmod = FlatLambdaCDM(name='Concordance',H0=70.0,Om0=0.3,Tcmb0=2.725, Neff=3.04, Ob0=0.0463)
pixscale = 0.000138889*3600. # arcsec (seems to be same for all cubes)
pixscale = pixscale/3600.*np.pi/180. # radians
comd_bax = cosmod.comoving_distance(Y_info['redshift']).value*1.0e3 #comoving dist in kpc
spaxsize_bax = pixscale*comd_bax/(1.0 + Y_info['redshift']) # kpc
spaxsize_bax = spaxsize_bax**2. # kpc^2
Y_info['logMS_dens_kpc'] = Y_info['logMs_dens'] - np.log10(spaxsize_bax)  # solar mass per kpc^2
Y_info['sSFR']=Y_info['logSFR_dens']-Y_info['logMS_dens_kpc']
#-----------------------------------------------------    

heightmap=np.zeros(map_size[0]*map_size[1])

#prep colours
finite_mask=(np.isfinite(Y_info[ATTR]))
vmin=Y_info[ATTR][finite_mask].min()
vmax=Y_info[ATTR][finite_mask].max()
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.coolwarm_r
m = cm.ScalarMappable(norm=norm, cmap=cmap)

colours=np.zeros((map_size[0]*map_size[1],test_gal.shape[1],4))
for bmu in range(map_size[0]*map_size[1]):
    masked=(Y_info['bmu']==int(bmu))
    temp_list=Y_info[ATTR][masked]
    temp_list.sort()
    for i in range(len(temp_list)):
        temp_spacing=int(4544/(len(temp_list)))
        colours[bmu,temp_spacing*i:temp_spacing*i+temp_spacing]=m.to_rgba(temp_list[i])
    heightmap[bmu]=len(Y_info[masked])

# heightmap normalized for background colouring
scaled_alpha=MinMaxScaler().fit_transform(heightmap.reshape(-1,1)).flatten()
scaled_alpha=scaled_alpha*0.6

# start the plotting
fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(12,12))
fig.suptitle(str('MaNGA_ID:'+ Y_info['mangaID'][0]))
#
decoded_prototypes = som.decode(som.prototypes) #decoded_prototypes are used in each cell
for k in range(map_size[0]*map_size[1]):
    # select prototype:
    x = decoded_prototypes[k]
    if sum(colours[k,0,:])!=0.:
        # create scatter for activated nodes, coloured by sorted values:
        ax[k // map_size[1]][k % map_size[1]].scatter(range(4544),x,c=colours[k,:,:],cmap=cmap,s=0.5)
        ax[k // map_size[1]][k % map_size[1]].set_xticks([])
        ax[k // map_size[1]][k % map_size[1]].set_yticks([])
    else:
        # if no activation, create ghost prototype:
        ax[k // map_size[1]][k % map_size[1]].plot(x,c='grey',alpha=0.15)
        ax[k // map_size[1]][k % map_size[1]].axis('off')
        # background for heightmap info:
    background = (decoded_prototypes[k].max()-decoded_prototypes[k].min())*np.random.random_sample(4544)+decoded_prototypes[k].min()
    ax[k // map_size[1]][k % map_size[1]].plot(background,alpha=scaled_alpha[k],c='grey') #background for activated
    ax[k // map_size[1]][k % map_size[1]].plot(background,alpha=0.015,c='black') #floor value of backgrounds
plt.subplots_adjust(hspace=1.01, wspace=1.01)

# 'colorbar'
gradient=(np.linspace(vmin,vmax,256))
gradient = np.vstack((gradient, gradient))
#
caxleft=plt.axes([0.1,0.05,0,0])
caxleft.text(0,0,str(round(vmin,8)))
caxleft.axis('off')
#
caxright=plt.axes([0.9,0.05,0,0])
caxright.text(0,0,str(round(vmax,8)),horizontalalignment='right')
caxright.axis('off')
#
caxcent=plt.axes([0.5,0.05,0,0])
caxcent.text(0,0,str(ATTR),horizontalalignment='center')
caxcent.axis('off')    
# left,bottom,weight,height
cax1=plt.axes([0.1,0.05,0.8,0.05])
cax1.imshow(gradient, cmap=cmap)
cax1.axis('off')
