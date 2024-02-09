# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils import data
import secrets

class BaseDataset(data.Dataset):
    def __init__(self, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):
        """This function initializes the parameters used for image processing.
        Parameters:
            - ignore_label (int): Label to be ignored during processing.
            - base_size (int): Base size of the image.
            - crop_size (tuple): Tuple of height and width for cropping the image.
            - downsample_rate (int): Rate at which the image is downsampled.
            - scale_factor (int): Factor by which the image is scaled.
            - mean (list): List of mean values for normalizing the image.
            - std (list): List of standard deviation values for normalizing the image.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Sets the base size, crop size, and ignore label for the image.
            - Sets the mean and standard deviation values for normalizing the image.
            - Sets the scale factor and downsample rate for processing the image.
            - Creates an empty list for storing the image files."""
        

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1./downsample_rate

        self.files = []

    def __len__(self):
        """"Returns the length of the files attribute."
        Parameters:
            - self (object): The object whose files attribute will be evaluated.
        Returns:
            - int: The length of the files attribute.
        Processing Logic:
            - Returns the length of the files attribute.
            - Uses the built-in len() function.
            - Does not modify the object.
            - Returns an integer value."""
        
        return len(self.files)
    
    def input_transform(self, image):
        """Transforms the input image by converting it to float32, inverting the color channels, normalizing it, and subtracting the mean and dividing by the standard deviation.
        Parameters:
            - image (numpy array): The input image to be transformed.
        Returns:
            - image (numpy array): The transformed image.
        Processing Logic:
            - Convert to float32.
            - Invert color channels.
            - Normalize.
            - Subtract mean.
            - Divide by standard deviation."""
        
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image
    
    def label_transform(self, label):
        """Transforms the label into an array of type int32.
        Parameters:
            - label (list): A list of labels to be transformed.
        Returns:
            - np.array: An array of type int32.
        Processing Logic:
            - Transform label into array.
            - Cast array to type int32."""
        
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, size, padvalue):
        """Pads an image with a specified size and pad value.
        Parameters:
            - image (numpy array): The image to be padded.
            - h (int): The height of the image.
            - w (int): The width of the image.
            - size (tuple): The desired size of the padded image in the format (height, width).
            - padvalue (int): The value to be used for padding.
        Returns:
            - numpy array: The padded image.
        Processing Logic:
            - Copies the original image.
            - Calculates the necessary padding for the image based on the desired size.
            - If padding is needed, uses the cv2.copyMakeBorder function to add padding to the image.
            - Returns the padded image."""
        
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=padvalue)
        
        return pad_image

    def rand_crop(self, image, label):
        """Docstring:
        Crops the given image and label to the specified crop size.
        Parameters:
            - image (numpy array): The image to be cropped.
            - label (numpy array): The label to be cropped.
        Returns:
            - image (numpy array): The cropped image.
            - label (numpy array): The cropped label.
        Processing Logic:
            - Pad the image and label to the specified crop size.
            - Generate random coordinates within the padded image.
            - Crop the image and label using the generated coordinates."""
        
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                                (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                                (self.ignore_label,))
        
        new_h, new_w = label.shape
        x = secrets.SystemRandom().randint(0, new_w - self.crop_size[1])
        y = secrets.SystemRandom().randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def center_crop(self, image, label):
        """Crops the center of an image and its corresponding label to a specified size.
        Parameters:
            - image (numpy array): The image to be cropped.
            - label (numpy array): The label corresponding to the image.
        Returns:
            - image (numpy array): The cropped image.
            - label (numpy array): The cropped label.
        Processing Logic:
            - Calculate the center coordinates of the image.
            - Crop the image and label using the calculated coordinates.
            - Return the cropped image and label."""
        
        h, w = image.shape[:2]
        x = int(round((w - self.crop_size[1]) / 2.))
        y = int(round((h - self.crop_size[0]) / 2.))
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label
    
    def image_resize(self, image, long_size, label=None):
        """Resizes an image to a specified long size while maintaining aspect ratio.
        Parameters:
            - image (numpy array): The image to be resized.
            - long_size (int): The desired long size of the image.
            - label (numpy array, optional): The label associated with the image. Defaults to None.
        Returns:
            - image (numpy array): The resized image.
            - label (numpy array): The resized label, if applicable.
        Processing Logic:
            - Calculate new height and width based on the long size and original aspect ratio.
            - Resize the image using the calculated dimensions and linear interpolation.
            - If a label is provided, resize it using the same dimensions and nearest neighbor interpolation.
            - If no label is provided, return only the resized image."""
        
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)
        
        image = cv2.resize(image, (new_w, new_h), 
                           interpolation = cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), 
                           interpolation = cv2.INTER_NEAREST)
        else:
            return image
        
        return image, label

    def multi_scale_aug(self, image, label=None, 
            rand_scale=1, rand_crop=True):
        """This function performs multi-scale augmentation on an image and its corresponding label, if provided.
        Parameters:
            - image (numpy array): The input image to be augmented.
            - label (numpy array, optional): The corresponding label for the input image. Defaults to None.
            - rand_scale (float, optional): The random scale factor to be applied to the base size. Defaults to 1.
            - rand_crop (bool, optional): Whether to perform random cropping on the augmented image. Defaults to True.
        Returns:
            - image (numpy array): The augmented image.
            - label (numpy array, optional): The augmented label, if provided.
        Processing Logic:
            - Calculates the long size of the image based on the base size and the random scale factor.
            - If a label is provided, resizes both the image and label to the long size.
            - If random cropping is enabled, performs random cropping on the augmented image and label.
            - Returns the augmented image and label, if provided. Otherwise, returns only the augmented image."""
        
        long_size = np.int(self.base_size * rand_scale + 0.5)
        if label is not None:
            image, label = self.image_resize(image, long_size, label)
            if rand_crop:
                image, label = self.rand_crop(image, label)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def gen_sample(self, image, label, 
            multi_scale=True, is_flip=True, center_crop_test=False):
        """"""
        
        if multi_scale:
            rand_scale = 0.5 + secrets.SystemRandom().randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, 
                                                    rand_scale=rand_scale)

        if center_crop_test:
            image, label = self.image_resize(image, 
                                             self.base_size,
                                             label)
            image, label = self.center_crop(image, label)

        image = self.input_transform(image)
        label = self.label_transform(label)
        
        image = image.transpose((2, 0, 1))
        
        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(label, 
                               None, 
                               fx=self.downsample_rate,
                               fy=self.downsample_rate, 
                               interpolation=cv2.INTER_NEAREST)

        return image, label

    def inference(self, model, image, flip=False):
        """"""
        
        size = image.size()
        pred, _, _ = model(x=image, train_step=-1)
        pred = F.upsample(input=pred, 
                          size=(size[-2], size[-1]), 
                          mode='bilinear')        
        if flip:
            flip_img = image.numpy()[:,:,:,::-1]
            flip_output, _, _ = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear')
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        """"""
        
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        device = torch.device("cuda:%d" % model.device_ids[0])
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 2.0 / 3.0)
        stride_w = np.int(self.crop_size[1] * 2.0 / 3.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).to(device)
        padvalue = -1.0  * np.array(self.mean) / np.array(self.std)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if max(height, width) <= np.min(self.crop_size):
                new_img = self.pad_image(new_img, height, width, 
                                    self.crop_size, padvalue)
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                if height < self.crop_size[0] or width < self.crop_size[1]:
                    new_img = self.pad_image(new_img, height, width, 
                                        self.crop_size, padvalue)
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).to(device)
                count = torch.zeros([1,1, new_h, new_w]).to(device)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        if h1 == new_h or w1 == new_w:
                            crop_img = self.pad_image(crop_img, 
                                                      h1-h0, 
                                                      w1-w0, 
                                                      self.crop_size, 
                                                      padvalue)
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)

                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred
