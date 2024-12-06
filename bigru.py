import os
import scipy.io as io
import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, GRU, Dense, Dropout
from keras.optimizers import RMSprop, AdamW, SGD
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras import layers, Model, models
from typing import Tuple
import math




# Set random seeds for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Print TensorFlow version for reference
print(f"TensorFlow version: {tf.__version__}")


file_ind = ['320','340','360','380','400','420','440']

Fs=3000
st=0.02 #stationary interval in terms of second
L=60 #block length


#%% divide between train, validation, and test

data_train_list = []
data_valid_list = []
data_test_list = []
import numpy as np
for file in file_ind:

#   there is a file for each subject in erp-data folder named by the following format: subject1.mat
    f = io.loadmat('NoLoad_'+file+'V.mat')
#   go through matlab maps to get the data
    a = float(file)*np.ones((len(f['Data1_AI_0']), 1))
    b = np.double(f['Data1_AI_0'])
    N=len(b)
    I=np.floor(N/L)-1  #total number of observations (N/L)
    Ntest=int(np.floor(I/4))   #we set 1/4 of I for test
    Nvalid = int(np.floor(3*I/16)) #validation is 1/4 of the 3/4*I (training) = 3/16
    Ntrain=int(I-Nvalid-Ntest)
    train_ind_max = Ntrain*L
    valid_ind_max = train_ind_max+Nvalid*L
    test_ind_max = valid_ind_max+Ntest*L

    data_temp_train = np.concatenate((a[0:train_ind_max], b[0:train_ind_max]), axis=1)
    data_temp_valid = np.concatenate((a[train_ind_max:valid_ind_max], b[train_ind_max:valid_ind_max]), axis=1)
    data_temp_test = np.concatenate((a[valid_ind_max:test_ind_max], b[valid_ind_max:test_ind_max]), axis=1)
    data_train_list.append(data_temp_train)
    data_valid_list.append(data_temp_valid)
    data_test_list.append(data_temp_test)

data_train = np.concatenate(data_train_list, axis=0) #convert the list to np arrays
data_valid = np.concatenate(data_valid_list, axis=0)
data_test = np.concatenate(data_test_list, axis=0)


#%% Normalize using mean and std of training

dmin=data_train.min(axis=0)
dmax=data_train.max(axis=0)
max_min = dmax - dmin
data_train  = (data_train-dmin)/max_min
data_valid  = (data_valid-dmin)/max_min
data_test  = (data_test-dmin)/max_min


#%% data generator

window = L
step = 1
delay = 0
batch_size = 64

def generator(data, window, delay, min_index, max_index,
              shuffle=False, batch_size=batch_size, step=step):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + window

    while 1:
        if shuffle:
            sample_ind = np.random.randint(
                    min_index, max_index//window, size=batch_size)
            rows = sample_ind*window
        else:
            if i >= max_index:
                    i = min_index + window
            rows = np.arange(i, min(i + batch_size*window, max_index), window)
            i = rows[-1]+window
        samples = np.zeros((len(rows),
                            window // step,
                            (data.shape[-1]-1))) #second argument is the number of time stamps (1440/6)
                            # first argument is number of samples (indepedent)
                            #third arg is e.g. 12 (size of features)
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - window, rows[j], step)
            samples[j] = data[indices,1:] #indexing reducing dimenion but slicing doesn't
            #d_min = np.amin(data[indices,1:],axis=0)
            #samples[j] = (data[indices,1:]-d_min)/(np.amax(data[indices,1:],axis=0)-d_min)
            #d_min = np.amin(data[indices,1:],axis=0)
            targets[j] = data[rows[j]-1 + delay][0] #target is the first column
        yield samples, targets



#%%
train_gen = generator(data_train,
                      window=window,
                      delay=delay,
                      min_index=0,
                      max_index=None,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size) #see what None does
#%%
val_gen = generator(data_valid,
                    window=window,
                    delay=delay,
                    min_index=0,
                    max_index=None,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size)

#%%
test_gen = generator(data_test,
                    window=window,
                    delay=delay,
                    min_index=0,
                    max_index=None,
                    step=step,
                    batch_size=batch_size)

#%%
val_steps = data_valid.shape[0]//(window*batch_size)
test_steps = data_test.shape[0]//(window*batch_size)



#import numpy as np
def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]






def mhsa_bigru(input_shape: Tuple[int, ...], bigru_units: int, num_heads: int, head_dim: int, num_layers: int) -> model.Model:
    """
    Constructs a model with stacked BiGRU layers followed by a Multi-Head Self-Attention (MHSA) layer.

    Args:
        input_shape (Tuple[int, ...]): Shape of the input data (e.g., (60, 1)).
        bigru_units (int): Number of units in each BiGRU layer. Default is 32.
        num_heads (int): Number of attention heads for the MHSA layer. Default is 4.
        num_layers (int): Number of stacked BiGRU layers. Default is 5.

    Returns:
        models.Model: A compiled Keras model.
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Stack BiGRU layers dynamically based on num_layers
    x = inputs
    for _ in range(num_layers):
        x = layers.Bidirectional(layers.GRU(bigru_units, return_sequences=True))(x)

    # Multi-Head Self-Attention layer
    mhsa = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(x, x)

    # Flatten the MHSA output
    flatten = layers.Flatten()(mhsa)

    # Output layer (single neuron for regression)
    outputs = layers.Dense(1)(flatten)

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# Example usage
input_shape = (60, 1)
model = mhsa_bigru(input_shape, bigru_units, num_heads, head_dim, num_layers)
model.summary()




filenameFig = ''
filenameFig = filenameFig + '_'
filename = filenameFig
filename_model = filename + '.h5'

model.summary()

model.compile(optimizer=AdamW(learning_rate=0.001), loss='mse', metrics=['mae','mape'])
history = model.fit(train_gen,
                          steps_per_epoch=500,
                          epochs=150,
                          validation_data=val_gen,
                          validation_steps=val_steps)


model.save(filename + '.keras')

import pickle
with open(filename, 'wb') as handle:
    pickle.dump(history.history, handle)

data_test_for_evaluate = data_valid[:,1:].reshape((len(data_valid)//window, window, 1))
targets_test = data_valid[:,:1].reshape((len(data_valid)//window, window, 1))
sample = np.zeros((1, window // step,
                            (data_valid.shape[-1]-1)))
predicted_targets = np.zeros((len(data_test_for_evaluate),))
true_targets = np.zeros((len(data_test_for_evaluate),))

for i in range(0,len(data_test_for_evaluate)):
    true_targets[i] = targets_test[i,window-1]
target_mean = true_targets.mean(axis=0)

for i in range(0,len(data_test_for_evaluate)):
    sample[0] = data_test_for_evaluate[i,]
    predicted_targets[i]=model.predict(sample)

MSE = sum(abs(predicted_targets-true_targets)**2)/len(true_targets)
MAE = sum(abs(predicted_targets-true_targets))/len(true_targets)

RRSE = 100 * np.sqrt(MSE * len(true_targets) / (sum(abs(true_targets-target_mean)**2)))
RAE = 100 * MAE * len(true_targets) / sum(abs(true_targets-target_mean))

print('MSE: ', MSE)
print('MAE: ', MAE)
print('RRSE: ', RRSE)
print('RAE: ', RAE)
print('target_mean: ', target_mean)
print('len(true_targets): ', len(true_targets))
print(sum(abs(true_targets-target_mean)**2))
print(sum(abs(true_targets-target_mean))/len(true_targets))
#plot
fig=plt.figure()
ax = fig.add_subplot(111)
# if we would like to read from a saved "history"

epoch_count = range(1, len(history.history['loss']) + 1)
#plt.plot(epoch_count, np.array(d['loss']), 'b--', labe$\mathit{M}$=$\mathit{L}$='training MAE')
#plt.plot(epoch_count, np.array(d['val_loss']), 'r-', labe$\mathit{M}$=$\mathit{L}$='validation MAE')
plt.plot(epoch_count, np.array(history.history['loss']), 'b--')
plt.plot(epoch_count, np.array(history.history['val_loss']), 'r-')
y=history.history['val_loss']
ymin = min(y)
xpos = y.index(min(y))
xmin = epoch_count[xpos]
y=history.history['val_mae']
yymin = min(y)

print('MSE by formula: ', MSE, ' MSE by model: ', ymin)

string1 = 'MSE = ' + '%.2E' % float(ymin)
string2 = '\n' + 'RAE = ' + to_str(round(RAE,2)) + '%' + '\n' + 'RRSE = ' + to_str(round(RRSE,2)) + '%'
string = string1 + string2
ax.annotate(string, xy=(xmin, ymin),xycoords='data',
              xytext=(-80, 85), textcoords='offset points',
                 bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
                 size=12,
                 arrowprops=dict(arrowstyle="->"))
plt.title('')
#xint = range(min(epoch_count), 15,2)
xint = range(min(epoch_count)-1, math.ceil(max(epoch_count)),20)
plt.xticks(xint)
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend(loc="best")
filename1 = filename + '_loss'
fig.set_size_inches(5.46, 3.83)
fig.savefig(filename1 + '.pdf', bbox_inches='tight')


#1st element of score: MSE (keras)
#2nd element of score: MSE
#3nd element of score: MAE
#4th element of score: RRSE
#5th element of score: RAE
score = []
score.append(ymin)
score.append(MSE)
score.append(MAE)
score.append(RRSE)
score.append(RAE)
filenameTXT = filename + '.txt'
np.savetxt(filenameTXT, score)

K.clear_session()
del model






