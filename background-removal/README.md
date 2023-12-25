An example of background removal including details using image matting.

## Key motivations:

- To remove background from profile pictures, I want to customize it with
one generated from Stable Diffusion.

## Key learnings:
- What's a trimap vs a simple mask.
- Kernel size is important.
- Matrix operations
- Pytorch
- Image Matting

## Key codes:

```
k_size=(8,8)
iterations=5

# for cv2.erode / dilate , k_size and interations
# are important; with lower values I get a pymatting error :
# ValueError: Conjugate gradient descent did not converge within 10000 iterations
```

```
trimap = np.full(mask.shape, 128)
trimap[eroded >= 1] = 255
trimap[dilated == 0] = 0

# eroded/dilated should be always compared to 0/1
# for binary masks
```

## Setup

```
conda create --name pymatting_env python=3.8
pip install
```
A GPU is optional, if nvidia is found, make sure to install
```
nvidia-container-toolkit
```


## Knowledge source:
- ChatGPT
- https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/  
- https://www.researchgate.net/figure/Morphological-operations-erosion-and-dilation-to-generate-the-trimap_fig5_270280273
- https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/
- https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
- https://github.com/PeterL1n/BackgroundMattingV2
- https://stackoverflow.com/questions/71837896/how-to-mask-outside-or-inside-an-arbitrary-shape-in-python
- https://xaviergeerinck.com/2022/08/16/image-matting/
