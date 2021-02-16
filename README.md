# TensorBoard-Embedding-Flowers

## Execution

Execute feature_extractor.py - it creates all the feature vectors that the projector will use for painting the clusters.

```bash
pỳthon feature_extractor.py --m X
```

"X" is a number between 1 to 4 which are identificators for the different models from imagenet: 

            1:mobilenet_v2_140_224
            2:inception_v3 
            3:resnet_v2_50 
            4:inception_resnet_v2
```python
if m == "1":  # feature extractor from model mobilenet_v2_140_224
    URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"
elif m == "2":  # feature extractor from model inception_v3
    URL = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3"
elif m == "3":  # feature extractor from model resnet_v2_50
    URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1"
elif m == "4":  # feature extractor from model inception_resnet_v2
    URL = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/3"
```


Open a terminal in the projector folder where there is another folder named logdir. Type the next command:

```bash
tensorboard --logdir = logdir
```

Logdir has the metadata file needed for coloring the images by labels, the projector configuration file and the 
spritesheet.

## After deploying the projector

Once the projector is deployed you need to adjust the options marked in red circles in the image below.

![Projector home menu](https://github.com/amgp-upm/cluster_visualization/blob/main/menu.png?raw=true)
