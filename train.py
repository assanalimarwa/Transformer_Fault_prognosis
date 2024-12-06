
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
