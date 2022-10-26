import segmentation_models as sm
from utils import load_data, ConstructDataset, Dataloder
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

## 1. Load Best Model
BACKBONE = 'efficientnetb3'
CLASSES = ['lip']
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

print("Load Best trained Model.")
sm.set_framework('tf.keras')
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, input_shape=(512, 512, 3))
# define optimizer
optim = tf.keras.optimizers.Adam(0.0001)
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)
model.load_weights('result/best_model.h5')
print("Successfully loaded!!")

## 2. Load Test Dataset
test_image_dir = "data/test/image"
test_mask_dir = "data/test/mask"
preprocess_input = sm.get_preprocessing(BACKBONE)

print('Loading data...')
if test_mask_dir is not None:
    test_dict = load_data(test_image_dir, test_mask_dir)
else:
    test_dict = load_data(test_image_dir, exist_mask = False)
    test_dict['masks'] = test_dict['images']

print("Test Image Shape: {}".format(test_dict['images'].shape))
test_dataset = ConstructDataset(
    test_dict['images'],
    test_dict['masks'],
    augmentation = None,
    preprocessing=preprocess_input
)
test_dataset_raw = ConstructDataset(
    test_dict['images'],
    test_dict['masks'],
    augmentation = None,
    preprocessing= None
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)
print("Successfully loaded!!")

## 3. Testing
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
scores = model.evaluate_generator(test_dataloader)

if test_mask_dir is not None:
    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))

## 4. Visualize
def visualize(index, **images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig('result/' + str(index) + '_result.png')


for i in range(len(test_dataset)):
    image, gt_mask = test_dataset[i]
    raw_image, raw_gt_mask = test_dataset_raw[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).round()
    if test_mask_dir is not None:
        visualize(
            image=raw_image.squeeze(),
            gt_mask=gt_mask.squeeze(),
            pr_mask=pr_mask.squeeze() * 255,
            index = i
        )
    else:
        visualize(
            image=raw_image.squeeze(),
            pr_mask=pr_mask.squeeze()* 255,
            index = i
        )