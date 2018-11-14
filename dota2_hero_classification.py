import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

# Membaca file CSV
sf_train = pd.read_csv('training_data.csv')

# Correlation Matrix untuk target features
corr_matrix = sf_train.corr()
print(corr_matrix['type'])

# Menghapus kolom yang tidak dibutuhkan
sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)
print(sf_train.head())

# Membaca file CSV data testing
sf_val = pd.read_csv('testing_data.csv')

# Membuang kolom yang tidak dibutuhkan
sf_val.drop(sf_val.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)

# Mendapatkan nilai array dari file CSV
train_data = sf_train.values
val_data = sf_val.values

# Use columns 2 to last as Input
train_x = train_data[:,2:]
val_x = val_data[:,2:]

# Use columns 1 as Output/Target (One-Hot Encoding)
train_y = to_categorical( train_data[:,1] )
val_y = to_categorical( val_data[:,1] )

# Create Network
inputs = Input(shape=(16,))
h_layer = Dense(10, activation='sigmoid')(inputs)

# Softmax Activation for Multiclass Classification
outputs = Dense(3, activation='softmax')(h_layer)

model = Model(inputs=inputs, outputs=outputs)

# Optimizer / Update Rule
sgd = SGD(lr=0.001)

# Compile the model with Cross Entropy Loss
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and use validation data
model.fit(train_x, train_y, batch_size=16, epochs=5000, verbose=1, validation_data=(val_x, val_y))
#model.save_weights('weights.h5')

# Predict all Validation data
predict = model.predict(val_x)

# Visualize Prediction
df = pd.DataFrame(predict)
df.columns = [ 'Strength', 'Agility', 'Intelligent' ]
df.index = val_data[:,0]
print(df)