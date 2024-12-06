
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
    f = io.loadmat('/content/NoLoad_'+file+'V.mat')
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