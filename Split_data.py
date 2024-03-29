print('\n\n running select_spectra_refactored')
import numpy as np
from astropy.table import Table, vstack, Column
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

LOAD_PATH = '/project/RAW_DATA_PATH/' # the far away original path.
SAVE_PATH = 'data/' # a closer path for spectra and waves saved as astropy tables.


'''
PART 0: put spectra into a table: [fname,mangaID,raw]
        do the same with wavelengths .
'''
def label_raws():
  ''' load each chunk. (1 million spectra were divided into 10 chunks)
      put a quick label on each spectra and wave and reduce to float32
      '''
  for i in range(10):
    data = Table.read(LOAD_PATH + 'datatab'+str(i)+'.fits')
    spec = np.load(LOAD_PATH + 'spec'+str(i)+'.npy')
    new_table_spec = Table([data['fname'],data['mangaID'],np.float32(spec)],
                      names=('fname','mangaID','raw'))
    new_table_spec.write(SAVE_PATH + 'spec'+str(i)+'.fits')

    wave = np.load(LOAD_PATH + 'wave'+str(i)+'.npy')
    new_table_wave = Table([data['fname'],data['mangaID'],np.float32(wave)],
                      names=('fname','mangaID','wave'))
    new_table_wave.write(SAVE_PATH + 'wave'+str(i)+'.fits')

label_raws()



'''
PART 1: LOAD FILES, APPLY COLUMNS FOR MASKING
'''

def load_and_merge_spec():
  ''' load each chunk
      returns spectra'''
  spec_table = Table.read(SAVE_PATH+'spec0.fits')
  for i in range(1,10):
    temp_spec_table = Table.read(SAVE_PATH+'spec'+str(i)+'.fits')
    spec_table = vstack([spec_table,temp_spec_table])
  return spec_table

def load_and_merge_wave():
  ''' load each chunk
      returns wave'''
  wave_table = Table.read(SAVE_PATH+'wave0.fits')
  for i in range(1,10):
    temp_wave_table = Table.read(SAVE_PATH+'wave'+str(i)+'.fits')
    wave_table = vstack([wave_table,temp_wave_table])
  return wave_table

def load_and_merge_data():
    ''' load each chunk
        returns data'''
  data = Table.read(LOAD_PATH+'datatab0.fits')
  for i in range(1,10):
    temp_data = Table.read(LOAD_PATH+'datatab'+str(i)+'.fits')
    data=vstack([data,temp_data])
  return data

def mask_mins(spec_table, data, percentile_limit=0.5):
    '''
    labels spectra that have low flux values
    '''
    x_mins = spec_table['raw'].min(axis=1)
    min_threshold = np.percentile(x_mins,percentile_limit)
    print('min threshold',np.percentile(x_mins,percentile_limit))
    min_mask = (x_mins > min_threshold)
    data['low_flux_outlier'] = (min_mask != True)
    return data
    
def id_high_flux(spec_table, data, percentile_limit=0.5):
    '''
    labels spectra that have high flux values
    '''
    x_maxes = spec_table['raw'].max(axis=1)
    max_threshold = np.percentile(x_maxes,100-percentile_limit)
    print('max threshold',np.percentile(x_maxes,100-percentile_limit))
    max_mask = (x_maxes < max_threshold)
    data['high_flux'] = (max_mask != True)
    return data, max_threshold

EM_LINES = [108, 1554, 1603, 1453, 2973, 2989]
def exclude_em_lines(spec_table, data, em_lines_list, max_flux_threshold, width=3):
    '''
    labels spectra that have high flux outside of known emission lines.
    
    em_lines_list: the centers of the emission lines at rest (in terms of wavelength bins)
    width: em_line +/- width defines the region of acceptible emission line
    '''
    high_flux_raws = spec_table['raw']
    all_outlier_mask = (spec_table['raw'] > max_flux_threshold)
    for em_line in em_lines_list:
        all_outlier_mask[:, em_line-width : em_line+width] = 0
        
    temp_column = Column(all_outlier_mask.max(axis=1),dtype=bool, name='high_flux_outlier')
    data.add_column(temp_column)
    return data


my_spec = load_and_merge_spec() #returns a table of spectra
my_wave = load_and_merge_wave() #returns a table of waves

# identify low flux outliers
my_data = mask_mins(my_spec,
                    load_and_merge_data(),
                    percentile_limit = 0.5)
# identify high flux outliers
my_data, max_flux_threshold = id_high_flux(my_spec,
                                           my_data,
                                           percentile_limit = 0.5)
# identify high flux outliers that are NOT emission lines
my_data = exclude_em_lines(my_spec, my_data, EM_LINES, max_flux_threshold)


# clean up data with masking
#-> 'criteria':[min val, max val]
MASKS = {'ngoodpixels': [3900, None],
         'meansn2': [10, None],
        }

def select_by_param(data, param, internal_min=None, internal_max=None):
  ''' 
  generates a mask based on some parameter
  returns that mask
  (eg, a minimum of 3900 'ngoodpixels' is required for the spectrum to be considered)
  '''
  assert type(param) == str
  if internal_min != None: min_mask = (data[param] >= internal_min)
  else: min_mask = np.ones(len(data), dtype=bool)

  if internal_max != None: max_mask = (data[param] <= internal_max)
  else: max_mask = np.ones(len(data), dtype=bool)

  total_mask = min_mask * max_mask
  return total_mask

def apply_masks(data, mask_dict):
  '''
  adds a column to the data table indicating whether to
  mask off the spectrum from training and testing.
  '''
  total_mask = np.ones(len(data), dtype=bool)
  for criteria in mask_dict.keys():
    print(criteria,mask_dict[criteria])
    temp_mask = select_by_param(data, criteria,
                                internal_min=mask_dict[criteria][0], #min
                                internal_max=mask_dict[criteria][1]) #max
    print(criteria, 'mask removes:', temp_mask.sum())
    total_mask = total_mask * temp_mask
    print(' total_mask:', total_mask.sum())
    data['masked_out'] = (total_mask != True)

apply_masks(my_data, MASKS)




'''
PART 2: SPLIT DATA INTO TRAIN/TEST/UNUSED SETS. SAVE.
'''

TEST_SIZE = 300_000

def train_test_flags(data, outlier_mask):
  '''
  adds columns ['Train_set'] and ['Test_set'] to data table to divide up spectra
  inputs: data table, mask removing low quality spectra (based on above criteria)
  '''
  test_column = Column(np.zeros(len(data),dtype=bool), name='Test_set')
  coinflip = np.zeros(len(data[outlier_mask]), dtype=bool)
  coinflip[:TEST_SIZE] = True
  np.random.seed(0)
  np.random.shuffle(coinflip)
  test_column[outlier_mask] = coinflip

  train_column = Column(np.zeros(len(data), dtype=bool), name='Train_set')
  inv_coinflip = (coinflip != True)
  train_column[outlier_mask] = inv_coinflip

  data.add_column(train_column)
  data.add_column(test_column)
  return data


mask = (my_data['masked_out'] == False)
low_flux_outlier = (my_data['low_flux_outlier'] == False)
high_flux_outlier = (my_data['high_flux_outlier'] == False)
total_mask = mask * low_flux_outlier * high_flux_outlier

my_data = train_test_flags(my_data, total_mask)




SUFFIX = 'DATE'

# save train set
train_mask = (my_data['Train_set'] == True)
my_spec[train_mask].write('x_train1m_' + SUFFIX + '.fits') #save the spec_table
my_data[train_mask].write('y_train1m_' + SUFFIX + '.fits') #save the data
my_wave[train_mask].write('w_train1m_' + SUFFIX + '.fits') #save the wave

# save test set
test_mask = (my_data['Test_set'] == True)
my_spec[test_mask].write('x_test1m_' + SUFFIX + '.fits')
my_data[test_mask].write('y_test1m_' + SUFFIX + '.fits')
my_wave[test_mask].write('w_test1m_' + SUFFIX + '.fits')

# save unused set
unused_mask = (total_mask != True)
my_spec[unused_mask].write('x_unused1m_' + SUFFIX + '.fits')
my_data[unused_mask].write('y_unused1m_' + SUFFIX + '.fits')
my_wave[unused_mask].write('w_unused1m_' + SUFFIX + '.fits')


# log cuts
print('\n')
print('ngp:',(my_data['ngoodpixels']<3900).sum())
print('meansn2:',(my_data['meansn2']<10).sum())
print('low flux outlier:',(my_data['low_flux_outlier']==True).sum())
print('high flux:',(my_data['high_flux']==True).sum(),'(Not removed)')
print('high flux outlier:',(my_data['high_flux_outlier']==True).sum(),'(Removed)')

