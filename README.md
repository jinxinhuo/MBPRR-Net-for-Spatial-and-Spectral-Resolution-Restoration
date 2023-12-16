# A Restoration Scheme for Spatial and Spectral Resolution of Panchromatic Image Using Convolutional Neural Network (MBPRR-Net)
Official implementation of "A Restoration Scheme for Spatial and Spectral Resolution of Panchromatic Image Using Convolutional Neural Network" submitted to **IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS)** 2023

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Citation](#citation)

## Introduction
Remote sensing images are the product of information obtained by various sensors, and the higher the resolution of the image, the more information it contains. Therefore, improving the resolution of the remote sensing image is conducive to identify earth resources from the remote sensing image. We present a multiple branch panchromatic image resolution restoration network based on convolutional neural network to improve the spatial and spectral resolution of panchromatic image simultaneously, named MBPRR-Net. Specifically, we adopt multi-branch structure to extract abundant features, and utilize a feature channel mixing block to enhance the interaction of adjacent channels between features. Feature aggregation in our method is used to learn more effective features from each branch, then a cubic filter is utilized to enhance the aggregated features. After feature extraction, we use a recovery architecture to generate the final image. Moreover, we utilize image super-resolution to restore spatial resolution and image colorization to restore spectral resolution, so that we compare with some image colorization and super-resolution methods to verify the proposed method. Experiments show that the performance of our method is outstanding in terms of visual effects and objective evaluation metrics against some existing excellent image super-resolution and colorization methods.

## Train
1. Run 'MBPRR/model/main.py'. The parameters 'img_train_pan_path', 'img_train_ms_path', 'img_test_pan_path', 'img_test_ms_path' and 'mode' are required.
    ```bash
    python main.py --img_train_pan_path your_path --img_train_ms_path your_path --img_test_pan_path your_path --img_test_ms_path your_path --mode train
    ```

## Test
1. Run 'MBPRR/model/main.py'. The parameters 'img_test_pan_path', 'img_test_ms_path' and 'mode' are required.
    ```bash
    python main.py --img_test_pan_path your_path --img_test_ms_path your_path --mode test
    ```

## Citation
```


```
