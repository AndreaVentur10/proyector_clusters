# TensorBoard-Embedding-Flowers

**Usage**

Execute feature_vectors.py - it creates all the feature vectors that the projector will use for painting the clusters

Open a terminal in the projector folder where there is another folder named logdir. Type the next command:

`tensorboard --logdir = logdir`

Logdir has the metadata file needed for coloring the images by labels, the projector configuration file and the 
spritesheet.