import numpy as np
import albumentations as A
from tensorflow.keras.utils import Sequence


def get_non_spatial_augmentation():
    non_spatial_augmentation = [
        A.GaussianBlur(p=0.5),

        A.OneOf(
            [
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        )

    ]
    return A.Compose(non_spatial_augmentation)


def get_spatial_augmentation():
    spatial_augmentation = [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.VerticalFlip(p=0.5)
    ]
    return A.Compose(spatial_augmentation)

spatial_augmentation = get_spatial_augmentation()
non_spatial_augmentation = get_non_spatial_augmentation()
augmentation = dict({'spatial': spatial_augmentation, 'non_spatial': non_spatial_augmentation})

class ConstructDataset(Sequence):
    """
    #####edited version#####
    Args:

        images_array (np.array): 
        masks_array (np.array): 
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)


    original
    
    CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    ## need to change ( is 'unlabelled' required? )
    CLASSES = ['lip']
    
  
    def __init__(
            self,
            images_array,
            masks_array,
            augmentation=augmentation,
            preprocessing=None,
    ):
        self.ids = images_array.shape[0] # number of pictures in each image and mask
        self.images_fps = images_array
        self.masks_fps = masks_array
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    """

    #original code
    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    """

    def __getitem__(self, i):

        # read data

        image = self.images_fps[i,:,:,:]
        mask = self.masks_fps[i,:,:,:]
        # mask 3-channels to 1-channel (grayscale)
        mask = np.dot(mask, [0.299, 0.587, 0.114])
        # black: 0, white: 1
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        
           # apply augmentations
        if self.augmentation is not None:
            spatial = self.augmentation['spatial'](image=image, mask=mask)
            image = spatial['image']
            mask = spatial['mask']
            # image = self.augmentation['non_spatial'](image=np.float32(image))['image']
        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image)

        return image, mask


    #### need to change because ids is not the same as the example given
    def __len__(self):
        return self.ids

class Dataloder(Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integer number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)