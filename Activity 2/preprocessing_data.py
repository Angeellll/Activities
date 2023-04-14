import numpy as np
from sklearn import preprocessing

# Preprocessing Data
input_data = np.array([[ 5.1, -2.9,  3.3],
                       [-1.2,  7.8, -6.1],
                       [ 3.9,  0.4,  2.1],
                       [ 7.3, -9.9, -4.5],
                     ])

# Binarization
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print('\nThe binarized data are: \n', data_binarized)

# Mean Removal
print('\nBEFORE:')
print('Mean = ', input_data.mean(axis=0))
print('Standard Deviation = ', input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)

print('\nAFTER:')
print('Mean = ', data_scaled.mean(axis=0))
print('Standard Deviation = ', data_scaled.std(axis=0))

# Scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print('\nMin-max scaled data:\n', data_scaled_minmax)

# Normalization
data_normalized_L1 = preprocessing.normalize(input_data, norm="l1")
data_normalized_L2 = preprocessing.normalize(input_data, norm="l2")
print('\nL1 normalized data:\n', data_normalized_L1)
print('\nL2 normalized data:\n', data_normalized_L2)
