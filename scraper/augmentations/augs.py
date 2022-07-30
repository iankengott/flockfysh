import albumentations as A
import cv2

horizontal_flip_transform = A.Compose([
    A.HorizontalFlip(always_apply=True),
], bbox_params = A.BboxParams(format="yolo"))

vertical_flip_transform = A.Compose([
    A.VerticalFlip(always_apply=True),
], bbox_params= A.BboxParams(format="yolo"))

pixel_dropout_transform = A.Compose([
    A.PixelDropout(always_apply=True),
], bbox_params = A.BboxParams(format="yolo"))


random_rotate = A.Compose([
    A.Rotate(always_apply=True),
], bbox_params = A.BboxParams(format="yolo") )

#NOTE: THIS METHOD IMPLIES THAT THE IMAGE WIDTHS MUST BE AT LEAST 50 PIXELS
#Remove this aug to remove this constraint
random_crop = A.Compose([
    A.RandomCrop(always_apply=True, width=50, height=50),
], bbox_params = A.BboxParams(format="yolo"))

augs = [horizontal_flip_transform, vertical_flip_transform, pixel_dropout_transform, random_rotate, random_crop]
def get_augmentations():
    return augs

def get_num_augmentations():
    return len(augs)