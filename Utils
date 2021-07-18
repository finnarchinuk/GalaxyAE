# A few functions used in multiple scripts


def normalize_4050_region(x_data, y_data, w_data):
    ''' Normalizes each spectrum by the 4050-4150 Angrstrom region.
    x_data: spectra
    y_data: physical parameters and spectra info
    w_data: wavelengths '''
    for i in range(len(x_data)):
        # deredshift wavelengths
        w_deredshifted = w_data['wave'][i] / (1+y_data['redshift'][i])

        # figure out mask
        mask4050 = (w_deredshifted > 4050)
        mask4150 = (w_deredshifted < 4150)
        window_mask = mask4050 * mask4150

        # get mean of subset
        temp_mean = np.mean(x_data['raw'][i][window_mask])

        # apply normalization to x_test
        x_data['raw'][i] = x_data['raw'][i] / temp_mean


def node2idx(index_int, mapsize=15):
    ''' Takes a node index (as an integer), returns an index (as a tuple).
    (Assumes a square map) '''
    row = int(index_int // mapsize)
    column = int(index_int % mapsize)
    return (column+1, mapsize-row)
    
    
def idx2node(index_tuple, mapsize=15):
    ''' Takes a node index tuple, returns the index as an integer
    (Assumes a square map) '''
    column_offset = (mapsize-index_tuple[1]) * mapsize
    row_offset = index_tuple[0]-1
    return column_offset + row_offset
