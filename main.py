from utils import load_data, ConstructDataset, Dataloder
import segmentation_models as sm
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os


## 0. Set parameters
train_image_dir = './data/train/sample_image_validating.h5'
train_mask_dir = './data/train/sample_mask_validating.h5'
valid_image_dir = './data/test/image'
valid_mask_dir = './data/test/mask'

os.makedirs('./result', exist_ok=True)

## 1. Load image, mask dataset
print('Loading data...')
train_dict = load_data(train_image_dir, mask_dir = train_mask_dir)
valid_dict = load_data(valid_image_dir, mask_dir = valid_mask_dir)
print("Train Image Shape: {}".format(train_dict['images'].shape))
print("Train Mask Shape: {}".format(train_dict['masks'].shape))
print("Valid Image Shape: {}".format(valid_dict['images'].shape))
print("Valid Mask Shape: {}".format(valid_dict['masks'].shape))
print('Done!')

## 2. Dataloader 생성
# Dataset for train images
BACKBONE = 'efficientnetb3'
BATCH_SIZE = 4
preprocess_input = sm.get_preprocessing(BACKBONE)

train_dataset = ConstructDataset(
    train_dict['images'],
    train_dict['masks'],
    preprocessing=preprocess_input
)

valid_dataset = ConstructDataset(
    valid_dict['images'],
    valid_dict['masks'],
    preprocessing=preprocess_input
)


# need to change, sample size is 100
train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

## 3. 모델 생성
CLASSES = ['lip']
LR = 0.0001
EPOCHS = 100 # need to change, it was 40

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
sm.set_framework('tf.keras')
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, input_shape=(512, 512, 3))

# define optimizer
optim = tf.keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./result/best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

## 4. 학습
# train model
history = model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)


# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('result/loss.png')
