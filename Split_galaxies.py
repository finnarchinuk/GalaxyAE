'''
Loads all data and spectra
Divide them into galaxies (creating 4609 files)

this should be run after 'Calc_BMU.py'.
'''


print('\n\n running split_galaxies.py')
import numpy as np
from astropy.table import Table, vstack
import time

# SPLIT BY MANGA_ID for each set of data
SAVE_PATH = 'data/gal_split_refactored/' #where spectra will be saved
LOAD_PATH = 'data/refactored/' #folder containing original data

SUFFIX = 'june22b'
DATA_SUFFIX = 'june22bmu' # (indicating the file has been properly updated using 'Calc_BMU.py')

'''
LOAD ALL THE DATA (AND SPECTRA AND WAVES)
'''
print('loading spectra, wave, and data')
my_data = vstack([Table.read('y_train1m_'+DATA_SUFFIX+'.fits'),
                  Table.read('y_test1m_'+DATA_SUFFIX+'.fits'),
                  Table.read('y_unused1m_'+DATA_SUFFIX+'.fits')])

my_spec = vstack([Table.read('x_train1m_'+SUFFIX+'.fits'),
                  Table.read('x_test1m_'+SUFFIX+'.fits'),
                  Table.read('x_unused1m_'+SUFFIX+'.fits')])

my_wave = vstack([Table.read('w_train1m_'+SUFFIX+'.fits'),
                  Table.read('w_test1m_'+SUFFIX+'.fits'),
                  Table.read('w_unused1m_'+SUFFIX+'.fits')])



def pull_unique_gal_indexes(data, save=False):
  ''' makes a list of all galaxies by their mangaID.
      return unique_ids, and unique_suffixes
      (useful in looping through galaxies)
      (a manga 'suffix' for object '1-23023' would just be '23023')
  '''
  unique_ids = np.unique(data['mangaID'])

  unique_suffixes = np.zeros(len(unique_ids))
  for i in range(len(unique_ids)):
    unique_suffixes[i]=unique_ids[i].split('-')[1]
  if save==True:
    np.save('gal_unique_indexes_june12.npy',unique_suffixes)
    print('saved gal_unique_indexes_june12.npy')
  else:
    return unique_ids, unique_suffixes

      
#    define_suffixes(my_data)
pull_unique_gal_indexes(my_data, save=True)
print('generated suffixes')      
      

def split_galaxies(spectra, data, wave):
  ''' goes through everything and filters out each galaxy.
  a data file is written for each galaxy with columns for spectra and wavelength values as columns.
  (This is done because only a few galaxies would ever be opened
  simultaneously and keeps everything together nicely.)
  '''
  gal_indexes, gal_suffixes = pull_unique_gal_indexes(data)
  for gal_id, gal_suffix in zip(gal_indexes, gal_suffixes):
    mask = (data['mangaID'] == gal_id)

    temp_spec = spectra[mask]['raw']
    temp_data = data[mask]
    temp_wave = wave[mask]['wave']

    temp_data['raw'] = temp_spec
    temp_data['wave'] = temp_wave
    temp_data.write(SAVE_PATH+str(int(gal_suffix))+'.fits')
  print('finished splitting galaxies')

split_galaxies(my_spec, my_data, my_wave)


