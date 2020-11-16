# GANscape
**Image inpainting** tool specialised for **landscape** pictures, utilizing the power of **deep learning** and **generative adversarial networks**. Developed for Deep Learning in Practice at **Budapest University of Technology and Economics** in Autumn 2020.

## Steps for compiling the databank
- We downloaded the **Places2** 256x256 train database from http://places2.csail.mit.edu/download.html
- Out of the about 1.8 million images, 244k were from categories, that we considered potentially useful.
- The YOLOv3 neural network was used to exclude images that contained people. We did this by distributing files into smaller folders, so that can be run on Colaboratory, but we used our CPU's as well. A batch of 10k images took about 4 hours to process.
- This left us with 203k good images, which we separated into train, test, and validation datasets.
- We used a 5% testing and a 10% validation split.
- An other script was used to cut out parts of the images. The original images, the cut out parts and the modified images were all saved, along with the coordinates of the cut out parts.
- Finally this left us with 3x203k images, with a total size of 5.4 gigabytes.

## Links for the databank

**Disclaimer**: We do not intend to distribute this database, we are only providing these for educational evaluation purposes.

Link for the data with the chosen classes:(not available anymore)
https://drive.google.com/file/d/15SSnj3CllTz_73jKHOVU-9HpYgRpxAYm/view?usp=sharing

Link for the data with only train data:
https://drive.google.com/file/d/1WGVtHwXxGNojM5xvq0EinaTX_q-o_rUA/view?usp=sharing

Link for the data with train, validation and test data:
https://drive.google.com/file/d/15Y-wFgWVc06hFLslSP_ABl3cr-LPMwOQ/view?usp=sharing

Link for the data with the 64x64 pictures:
https://drive.google.com/file/d/1mkQG3OeBWKsnZNwciwP6k0wGEAoVUMTM/view?usp=sharing

Link for the data with the 64x64 pictures that have 28x28 cropped parts:
https://drive.google.com/file/d/1_RrWfKYkiXOKfmU8UxtWFqMNqNKQtQGF/view?usp=sharing
