import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

# Membaca file CSV untuk data training
sf_train = pd.read_csv('training_data.csv')

# Menjadikan matrix korelasi untuk target csv
corr_matrix = sf_train.corr()
print(corr_matrix['type'])

# Membuang kolom yang tidak diperlukan
sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)
print(sf_train.head())

#%%

# Membaca file csv untuk data testing
sf_val = pd.read_csv('testing_data.csv')

# Membuang kolom yang tidak dibutuhkan
sf_val.drop(sf_val.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)

# Merubah data dari library panda menjadi data array
train_data = sf_train.values
val_data = sf_val.values

# Menggunakan kolom kedua sebagai data training dan testing
train_x = train_data[:,2:]
val_x = val_data[:,2:]

# Menggunakan kolom pertama untuk target output
train_y = to_categorical( train_data[:,1] )
val_y = to_categorical( val_data[:,1] )

#%%
# Membuat network
inputs = Input(shape=(16,))
h_layer = Dense(10, activation='sigmoid')(inputs)

# Softmax Activation untuk klasifikasi multi kelas
outputs = Dense(3, activation='softmax')(h_layer)

model = Model(inputs=inputs, outputs=outputs)

# Optimizer
sgd = SGD(lr=0.001)
#adam = Adam(lr=0.001)

# Compile model
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model menggunakan data testing
model.fit(train_x, train_y, batch_size=16, epochs=5000, verbose=1, validation_data=(val_x, val_y))
#model.save_weights('weights.h5')

# Memprediksi semua data testing
predict = model.predict(val_x)

# Menampilkan prediksi
df = pd.DataFrame(predict)
df.columns = [ 'Strength', 'Agility', 'Intelligent' ]
df.index = val_data[:,0]
print(df)