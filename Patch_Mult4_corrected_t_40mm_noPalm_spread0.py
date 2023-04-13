import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras import layers
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

'''
Setting the parameters of the transformer models
'''
img_shape = (256, 256, 3)
input_shape = (256, 256, 3)
num_classes = 7

learning_rate = 0.0005  # was 0.001
weight_decay = 0.00005  # was 0.0001
batch_size = 64
num_epochs = 200
image_size = 144  # was 72
patch_size = 18  # was 24
num_patches = ((image_size // (patch_size // 2)) - 1) ** 2
projection_dim = 128  # was 32
depth = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 16  # was 8
mlp_head_units = [4096, 2048]

'''
Load image_arr and num_arr to save time
Separate into training and testing sets
'''

data_path = "C:/Users/ST/PycharmProjects/patch_tactile_corrected_40mm/40mmspread0_correct/"

image_arr_train = np.load(data_path + "image_arr_train.npy")
num_arr_train = np.load(data_path + "num_arr_train.npy")
image_arr_val = np.load(data_path + "image_arr_val.npy")
num_arr_val = np.load(data_path + "num_arr_val.npy")
image_arr_test = np.load(data_path + "image_arr_test.npy")
num_arr_test = np.load(data_path + "num_arr_test.npy")

print(len(image_arr_train))

x_test = image_arr_test
y_test = num_arr_test

x_train = image_arr_train
y_train = num_arr_train

n, bins, patches = plt.hist(y_test)
plt.title('Distribution of real 40mm balls')
plt.xlabel("Number of balls")
plt.ylabel("Number of cases")
# plt.show()

print(len(y_test))
print(np.amax(y_test))
print(np.amin(y_test))

'''
keras data augmentation
'''
data_augmentation = keras.Sequential(
    [
        #  layers.Normalization(),
        layers.Resizing(image_size, image_size),
        #  layers.RandomBrightness(factor=0.2),
        layers.RandomContrast(factor=0.2),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name='data_augmentation',
)

'''
keras data augmentation
'''
data_augmentation2 = keras.Sequential(
    [
        #  layers.Normalization(),
        layers.Resizing(image_size, image_size),
        #  layers.RandomFlip("horizontal"),
        #  layers.RandomRotation(factor=0.02),
        #  layers.RandomZoom(
        #      height_factor=0.2, width_factor=0.2
        #  ),
    ],
    name='data_augmentation2',
)

'''
Multi-layer perceptron (mlp)
'''


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


'''
class Patches(layers.Layer)
- for making patches
Output:
- patches
'''


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        print('patch_size', patch_size)

    def call(self, input_img):
        sample_size = tf.shape(input_img)[0]
        print('tf.shape(input_img)', tf.shape(input_img))
        patches = tf.image.extract_patches(
            images=input_img,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size / 2, self.patch_size / 2, 1],
            # strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        # print('patch_dims', patch_dims)
        # print('patches.shape', patches.shape)
        patches = tf.reshape(patches, [sample_size, -1, patch_dims])
        # print('old patches.shape', patches.shape)

        return patches


plt.figure(figsize=(4, 4))
image = x_test[np.random.choice(range(x_test.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")

'''
class PatchEncoder
- for:
1) patch projection
2) position embedding layer
output = new_patch
'''


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        # print('num_patches', self.num_patches)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # print(len(positions))
        # print(len(self.projection(patch)))
        # print(len(self.position_embedding(positions)))
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


'''
function for the transformer
Output: model
'''


def vit_classifier_mod():
    inputs = layers.Input(shape=input_shape)
    print('inputs', inputs.shape)
    augmented = data_augmentation(inputs)
    print('augmented', augmented.shape)
    patches = Patches(patch_size)(augmented)
    print('patches', patches.shape)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    print('encoded_patches', encoded_patches.shape)

    for i in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        print('x1', x1.shape)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        print('x2', x2.shape)
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        print('x3', x3.shape)
        encoded_patches = layers.Add()([x3, x2])
        print('encoded_patches', encoded_patches.shape)

    classifier_out = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    classifier_out = layers.Flatten()(classifier_out)
    classifier_out = layers.Dropout(0.5)(classifier_out)

    features = mlp(classifier_out, hidden_units=mlp_head_units, dropout_rate=0.5)
    output_class = layers.Dense(num_classes, activation=tf.nn.softmax)(features)
    # output_class = layers.Dense(num_classes)(features)

    model = keras.Model(inputs=inputs, outputs=output_class)
    return model

# Fast version


def asymmetric_loss4(alpha):
    def loss(y_true, y_pred):

        df3 = tf.argmax(y_pred, 1, name=None)
        df4 = tf.argmax(y_true, 1, name=None)
        y_true_reshaped = tf.gather(y_true, df4, axis=1, batch_dims=1)
        delta = int(y_true_reshaped) - int(df3)

        select1 = tf.gather(y_pred, y_true_reshaped, axis=1, batch_dims=1) + 1e-20

        loss0a = float(-1 * (tf.math.log(select1)))
        loss0b = -1 * float(delta / 14) * float(tf.math.log(select1) * 0.2)
        loss0 = loss0a - loss0b

        return loss0

    return loss

'''
Compile, train, test
'''

# +ve to punish overestimation, between -1 and 1
# want to avoid underestimation, so use -ve

alpha = -0.5


def main(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=asymmetric_loss4(alpha=alpha),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    checkpoint_filepath = data_path + "asym4/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    model.load_weights(checkpoint_filepath)

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(image_arr_val, num_arr_val),
        callbacks=[checkpoint_callback],
    )

    return history


vit_classifier = vit_classifier_mod()
history = main(vit_classifier)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
images_dir = data_path + 'asym4/'
plt.savefig(f"{images_dir}/acc.jpg")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(f"{images_dir}/loss.jpg")
plt.show()

'''
Compile, test
'''

# +ve to punish overestimation, between -1 and 1
# want to avoid underestimation, so use -ve
alpha = -0.5


def main(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=asymmetric_loss4(alpha=alpha),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = data_path + "asym4/checkpoint"

    model.load_weights(checkpoint_filepath)

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    start = time.time()
    predictions = model.predict(x_test)
    end = time.time()
    print('prediction time = ', end - start)
    with open(data_path + 'asym4/predictions.csv', 'w') as w1:
        wfile = csv.writer(w1)
        wfile.writerows(predictions)
    print("Predictions: ", predictions)
    np.save(data_path + "asym4/predictions", predictions)


vit_classifier = vit_classifier_mod()
main(vit_classifier)

predictions = np.load(data_path + "asym4/predictions.npy")

df1 = predictions
df1 = df1.argmax(axis=1)
print('df1 max', max(df1))

cm = confusion_matrix(y_test, df1)

ax = plt.subplot()
sns.color_palette("YlOrBr", as_cmap=True)
sns.heatmap(cm, cmap="Reds", annot=True, fmt='g',
            ax=ax)  # annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix (Total = %s)' % len(y_test))
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5'])
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5'])

plt.show()

for m in range(cm.shape[0]):
    print('%s-ball: %s%%' % (m, int(round(100 * cm[m, m] / np.sum(cm[m, :]), 0))))
print('Overall: %s%%' % (int(round(100 * np.trace(cm) / np.sum(cm[:, :]), 0))))

err_arr0 = df1 - y_test

print(len(y_test))
print(len(df1))
print(len(err_arr0))

print(max(err_arr0), min(err_arr0))

n, bins, patches = plt.hist(err_arr0, bins=[-2, -1, 0, 1, 2, 3, 4], align='left', rwidth=0.5)

print(n)
print(bins)
print(patches)

plt.title('Distribution of prediction error (Predict - True)')
plt.xlabel("Error in # balls")
plt.ylabel("Number of cases")

for n, b in zip(n, bins):
    plt.gca().text(b - 0.1, n + 0.2, str(n))

plt.show()
