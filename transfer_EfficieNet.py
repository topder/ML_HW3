# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt

# %%

root = "./"
imglabel_map = os.path.join(root, 'imagelabels.mat')
setid_map = os.path.join(root, 'setid.mat')
imagelabels = sio.loadmat(imglabel_map)['labels'][0]
setids = sio.loadmat(setid_map)
ids = np.concatenate([setids['trnid'][0], setids['valid'][0],setids['tstid'][0]])
labels = []
image_path = []
for i in ids:
    labels.append(int(imagelabels[i-1])-1)
    image_path.append( os.path.join(root, 'jpg', 'image_{:05d}.jpg'.format(i)))

df = pd.DataFrame({'filename':image_path, 'label':labels})

# %%

df.label=df.label.astype(str)
df_train , df_test  = train_test_split(df, test_size=0.25, random_state=42,)
df_train, df_val  = train_test_split(df_train, test_size=0.3333, random_state=42)

len(df_train), len(df_val), len(df_test)

# Create the ImageDataGenerator object
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
) 

test_datagen = ImageDataGenerator(
) 

# Generate batches and augment the images
train_generator = train_datagen.flow_from_dataframe(
    df_train,
    directory='./',
    x_col='filename',
    y_col='label',
    class_mode='categorical',
    target_size=(224, 224),
)

val_generator = val_datagen.flow_from_dataframe(
    df_val,
    directory='./',
    x_col='filename',
    y_col='label',
    class_mode='categorical',
    target_size=(224, 224),
)

test_generator = test_datagen.flow_from_dataframe(
    df_test,
    directory='./',
    x_col='filename',
    y_col='label',
    class_mode='categorical',
    target_size=(224, 224),
    shuffle="False"
)


# %%
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import EfficientNetB0



feature_extractor = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3), classes=102)

feature_extractor.trainable = False
input_ = tf.keras.Input(shape=(224, 224, 3))
x = feature_extractor(input_, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output_ = tf.keras.layers.Dense(102, activation='softmax')(x)
model = tf.keras.Model(input_, output_)
# Compile it
model.compile(optimizer="adam",
              loss="CategoricalCrossentropy",
              metrics=['accuracy'])


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, model, test_generator):
        self.model = model
        self.test_generator = test_generator
        self.loss = []
        self.acc = []

    def on_epoch_end(self, epoch, logs={}):
        loss, acc= model.evaluate(test_generator,verbose=0)
        self.loss.append(loss)
        self.acc.append(acc)
        print(" ||Test: Loss: %.2f, Accuracy: %.2f" % (loss, acc))


callback=[CustomCallback(model, test_generator)]
history_pre_train= model.fit(train_generator, epochs=20, validation_data=val_generator,callbacks=callback)
test_pre_train= callback[0]

feature_extractor.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
              loss="CategoricalCrossentropy",
              metrics=['accuracy']
)
epochs = 10
callback=[CustomCallback(model, test_generator)]
history_fine_tune= model.fit(train_generator, epochs=epochs, validation_data=val_generator,callbacks=callback)
test_fine_tune= callback[0]


# %% [markdown]
# Second Run

# %%
df.label=df.label.astype(str)
df_train , df_test  = train_test_split(df, test_size=0.25, random_state=100,)
df_train, df_val  = train_test_split(df_train, test_size=0.3333, random_state=100)

len(df_train), len(df_val), len(df_test)

# Create the ImageDataGenerator object
train_datagen = ImageDataGenerator(
    # rescale = 1./255,

    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
        # rescale = 1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
) 

test_datagen = ImageDataGenerator(
        # rescale = 1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
) 

# Generate batches and augment the images
train_generator = train_datagen.flow_from_dataframe(
    df_train,
    directory='./',
    x_col='filename',
    y_col='label',
    class_mode='categorical',
    target_size=(224, 224),
)

val_generator = val_datagen.flow_from_dataframe(
    df_val,
    directory='./',
    x_col='filename',
    y_col='label',
    class_mode='categorical',
    target_size=(224, 224),
)

test_generator = test_datagen.flow_from_dataframe(
    df_test,
    directory='./',
    x_col='filename',
    y_col='label',
    class_mode='categorical',
    target_size=(224, 224),
    shuffle="False"
)


# %%
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import EfficientNetB0



feature_extractor = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3), classes=102)

feature_extractor.trainable = False
input_ = tf.keras.Input(shape=(224, 224, 3))
x = feature_extractor(input_, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output_ = tf.keras.layers.Dense(102, activation='softmax')(x)
model = tf.keras.Model(input_, output_)
# Compile it
model.compile(optimizer="adam",
              loss="CategoricalCrossentropy",
              metrics=['accuracy'])


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, model, test_generator):
        self.model = model
        self.test_generator = test_generator
        self.loss = []
        self.acc = []

    def on_epoch_end(self, epoch, logs={}):
        loss, acc= model.evaluate(test_generator,verbose=0)
        self.loss.append(loss)
        self.acc.append(acc)
        print(" ||Test: Loss: %.2f, Accuracy: %.2f" % (loss, acc))


callback=[CustomCallback(model, test_generator)]
history_pre_train_2= model.fit(train_generator, epochs=20, validation_data=val_generator,callbacks=callback)
test_pre_train_2= callback[0]

feature_extractor.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
              loss="CategoricalCrossentropy",
              metrics=['accuracy']
)
epochs = 10
callback=[CustomCallback(model, test_generator)]

history_fine_tune_2= model.fit(train_generator, epochs=epochs, validation_data=val_generator,callbacks=callback)
test_fine_train_2= callback[0]


# %%
test_pre_train

# %%
# df = pd.DataFrame.from_dict(history_pre_train.history, orient='index').T
# df = df.append(pd.DataFrame.from_dict(history_pre_train_2.history, orient='index').T)
# df["test_loss"]=test_pre_train+test_pre_train_2.loss
# # df["test_loss"]=test_pre_train.loss+test_pre_train_2.loss
# df["test_acc"]=test_pre_train+test_pre_train_2.acc
# df=df.reset_index().rename(columns={'index':'epoch'})

# %%
df = pd.DataFrame.from_dict(history_pre_train.history, orient='index').T
# df = df.append(pd.DataFrame.from_dict(history_pre_train.history, orient='index').T)
df = df.append(pd.DataFrame.from_dict(history_pre_train_2.history, orient='index').T)
df["test_loss"]=test_pre_train.loss+test_pre_train_2.loss
df["test_acc"]=test_pre_train.acc+test_pre_train_2.acc
df=df.reset_index().rename(columns={'index':'epoch'})
temp= df.melt(id_vars=['epoch'], value_vars=['loss', 'val_loss',"test_loss"], var_name='metric', value_name='value')
#change the value in colum
temp.loc[temp['metric'] == 'loss', 'metric'] = 'Training Loss'
temp.loc[temp['metric'] == 'val_loss', 'metric'] = 'Validation Loss'
temp.loc[temp['metric'] == 'test_loss', 'metric'] = 'Test Loss'
temp
sns.set_theme(style="darkgrid")
plt.figure(figsize=(20,10))
sns.lineplot(data=temp,x="epoch", y="value", hue="metric",).set(xlabel='Epoch', ylabel='Loss', title='Cross-Entropy Loss vs Epochs for EfficientNetB0 \n(without fine_tune)')
# save fig
plt.savefig('EfficientNetB0_without_fine_tune_loss_vs_epochs.png')


# %%
temp= df.melt(id_vars=['epoch'], value_vars=['accuracy', 'val_accuracy',"test_acc"], var_name='metric', value_name='value')
#change the value in colum
temp.loc[temp['metric'] == 'accuracy', 'metric'] = 'Training Accuracy'
temp.loc[temp['metric'] == 'val_accuracy', 'metric'] = 'Validation Accuracy'
temp.loc[temp['metric'] == 'test_acc', 'metric'] = 'Test Accuracy'
temp
sns.set_theme(style="darkgrid")
plt.figure(figsize=(20,10))
sns.lineplot(data=temp,x="epoch", y="value", hue="metric",).set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy vs Epochs for EfficientNetB0 \n(without fine_tune)')
#
# save fig
plt.savefig('EfficientNetB0_without_fine_tune_accuracy_vs_epochs.png')

# %%
df

# %%
history_fine_tune.history

# %%
df

# %%
df = pd.DataFrame.from_dict(history_fine_tune.history, orient='index').T
# df = df.append(pd.DataFrame.from_dict(history_pre_train.history, orient='index').T)
df = df.append(pd.DataFrame.from_dict(history_fine_tune_2.history, orient='index').T)
df["test_loss"]=test_fine_tune.loss+test_fine_train_2.loss

# df["test_loss"]=test_pre_train.loss+test_pre_train_2.loss
df["test_acc"]=test_fine_tune.acc+test_fine_train_2.acc
df=df.reset_index().rename(columns={'index':'epoch'})
temp= df.melt(id_vars=['epoch'], value_vars=['loss', 'val_loss',"test_loss"], var_name='metric', value_name='value')
#change the value in colum
temp.loc[temp['metric'] == 'loss', 'metric'] = 'Training Loss'
temp.loc[temp['metric'] == 'val_loss', 'metric'] = 'Validation Loss'
temp.loc[temp['metric'] == 'test_loss', 'metric'] = 'Test Loss'
temp
sns.set_theme(style="darkgrid")
plt.figure(figsize=(20,10))
sns.lineplot(data=temp,x="epoch", y="value", hue="metric",).set(xlabel='Epoch', ylabel='Loss', title='Cross-Entropy Loss vs Epochs for EfficientNetB0 \n(with fine_tune)')
# save fig
plt.savefig('EfficientNetB0_with_fine_tune_loss_vs_epochs.png')

# %%
temp= df.melt(id_vars=['epoch'], value_vars=['accuracy', 'val_accuracy',"test_acc"], var_name='metric', value_name='value')
#change the value in colum
temp.loc[temp['metric'] == 'accuracy', 'metric'] = 'Training Accuracy'
temp.loc[temp['metric'] == 'val_accuracy', 'metric'] = 'Validation Accuracy'
temp.loc[temp['metric'] == 'test_acc', 'metric'] = 'Test Accuracy'
temp
sns.set_theme(style="darkgrid")
plt.figure(figsize=(20,10))
sns.lineplot(data=temp,x="epoch", y="value", hue="metric",).set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy vs Epochs for EfficientNetB0 \n(with fine_tune)')
#
# save fig
plt.savefig('EfficientNetB0_with_fine_tune_accuracy_vs_epochs.png')



