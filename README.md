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

## Databank modifications in order to optimize the data for the network
- Having the 256x256 database created before, we use this as a strating point, and our frist step is to make it 64x64 so we could use data that require less space
- We set the crop sizes from 20x20 to 25x25 which means 10-15% of the area of the picure will be cropped
- Around these crops we set a unifrom 28x28 rectangle. This was important, because the one of the netwrok's inputs is the cropped part, therefore we need a part on every picture that has the same dimensions.
- The test and validation spilt stayed the same
- So at the this left us with 3x203k images, with a total size of 807 Megabytes

## dataloading class
In order to make the loading in of the pictures smooth, we made an individual class for this purpose. The DataRead class stores the needed data for the training. Further information about this is class is in the manuals folder

## Constructing the network
- While constructing the network we relied heavily on the article: http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf and also on the implementation of a netwrok based on this article: https://github.com/V1NAY8/glcic
- Our network has a GAN structure as it has a generator and a discriminator part
- The generator gets the cropped image and tries to generate the missing part od the picture
- The discriminator gets the generated picture and also individually the part that needed to generated, in our netwrok this is 64x64 picture and a  28x28 cropped part. With the given inputs the discriminator tries to determine whether the picture is generated by the generator or not
- The training of the network is made of three parts. First we train the generator with MSE loss, then we train the discriminator individually with generated and also not generated pictures with a binary crossentropy loss, and finally we train the generator and discriminator together with the discriminators weights locked with a joint loss, for the generator to be able to fool the discriminator.
- Scripts related to handling the network (create, train, test, help functions etc.) are in in `network_build_run.ipynb` jupyter notebook, and we have ran them at Google Colaboratory

## Links for the databank

**Disclaimer**: We do not intend to distribute this database, we are only providing these for educational evaluation purposes.

Link for the data with the chosen classes:(not available anymore)
https://drive.google.com/file/d/15SSnj3CllTz_73jKHOVU-9HpYgRpxAYm/view?usp=sharing

Link for the data with only train data:
https://drive.google.com/file/d/1WGVtHwXxGNojM5xvq0EinaTX_q-o_rUA/view?usp=sharing

Link for the data with train, validation and test data:
https://drive.google.com/file/d/15Y-wFgWVc06hFLslSP_ABl3cr-LPMwOQ/view?usp=sharing

Link for the data with the 64x64 pictures:(not available anymore)
https://drive.google.com/file/d/1mkQG3OeBWKsnZNwciwP6k0wGEAoVUMTM/view?usp=sharing

Link for the data with the 64x64 pictures that have 28x28 cropped parts:
https://drive.google.com/file/d/1_RrWfKYkiXOKfmU8UxtWFqMNqNKQtQGF/view?usp=sharing

Link for the data with the 64x64 pictures that have 28x28 cropped parts(fixed):
https://drive.google.com/file/d/1N9R-XnwEJuN2W6rURWloKTZdHW1UQcku/view?usp=sharing

## Training and validation log output (1 epoch)
```python
Training with 
  -epocsh=1,
  -batchSize=128
  -steps for training=1349,
  -steps for validation=158,
  -phase I until step=674,
 -phase II until step=1078

---Epoch 1 --- Step 100 --- PHASE I --- mse_loss=17754.4633984375
---Epoch 1 --- Step 200 --- PHASE I --- mse_loss=17681.245185546875
---Epoch 1 --- Step 300 --- PHASE I --- mse_loss=17591.102786458334
---Epoch 1 --- Step 400 --- PHASE I --- mse_loss=17524.984619140625
---Epoch 1 --- Step 500 --- PHASE I --- mse_loss=17438.8034296875
---Epoch 1 --- Step 600 --- PHASE I --- mse_loss=17367.893868815103
---Epoch 1 --- Step 674 --- PHASE I --- mse_loss=17298.845653862205
---Epoch 1 --- Step 700 --- PHASE II --- disc_adversarial_loss=1.116056623367163
---Epoch 1 --- Step 800 --- PHASE II --- disc_adversarial_loss=0.6800238821241591
---Epoch 1 --- Step 900 --- PHASE II --- disc_adversarial_loss=0.46966667229359127
---Epoch 1 --- Step 1000 --- PHASE II --- disc_adversarial_loss=0.35285670142277986
---Epoch 1 --- Step 1078 --- PHASE II --- disc_adversarial_loss=0.2939081736609782
---Epoch 1 --- Step 1100 --- PHASE III --- disc_adversarial_loss=0.0636080652475357 --- mse_loss=15936.828125 --- gen_adversarial_loss=7.32526969909668
---Epoch 1 --- Step 1200 --- PHASE III --- disc_adversarial_loss=0.13049277663230896 --- mse_loss=17505.869140625 --- gen_adversarial_loss=5.722434043884277
---Epoch 1 --- Step 1300 --- PHASE III --- disc_adversarial_loss=0.17044642567634583 --- mse_loss=17710.447265625 --- gen_adversarial_loss=5.182978630065918
---Epoch 1 --- Step 1349 --- PHASE III --- disc_adversarial_loss=0.17443254590034485 --- mse_loss=17454.470703125 --- gen_adversarial_loss=5.132567405700684
---Epoch 1 --- Step 30 --- VALIDATION --- mse_loss=16657.4021484375
                              --- disc_adversarial_loss=1.3973292549451193
                              --- mse_loss=16657.585546875 --- gen_adversarial_loss=1.8330340186754863
---Epoch 1 --- Step 60 --- VALIDATION --- mse_loss=16615.379069010418
                              --- disc_adversarial_loss=1.398790431022644
                              --- mse_loss=16615.56222330729 --- gen_adversarial_loss=1.83028893272082
---Epoch 1 --- Step 90 --- VALIDATION --- mse_loss=16621.01257595486
                              --- disc_adversarial_loss=1.3986530277464124
                              --- mse_loss=16621.19572482639 --- gen_adversarial_loss=1.8311513993475173
---Epoch 1 --- Step 120 --- VALIDATION --- mse_loss=16642.143147786457
                              --- disc_adversarial_loss=1.3977418760458629
                              --- mse_loss=16642.326529947917 --- gen_adversarial_loss=1.8331491986910502
---Epoch 1 --- Step 150 --- VALIDATION --- mse_loss=16634.331953125
                              --- disc_adversarial_loss=1.3992185052235921
                              --- mse_loss=16634.515071614584 --- gen_adversarial_loss=1.8304136872291565
---Epoch 1 --- Step 157 --- VALIDATION --- mse_loss=16626.687160057358
                              --- disc_adversarial_loss=1.399647294720517
                              --- mse_loss=16626.87022226068 --- gen_adversarial_loss=1.829859287678441
```

## Test log output
```python
Test with 
  -batchSize=128
  -steps=79

Step 20 --- TEST --- mse_loss=16764.0546875
                  --- disc_adversarial_loss=1.395849919319153
                  --- mse_loss=16764.238623046876 --- gen_adversarial_loss=1.8387678205966949
Step 40 --- TEST --- mse_loss=16695.228588867187
                  --- disc_adversarial_loss=1.3953976511955262
                  --- mse_loss=16695.412817382814 --- gen_adversarial_loss=1.8420817971229553
Step 60 --- TEST --- mse_loss=16705.770556640626
                  --- disc_adversarial_loss=1.3982006748517355
                  --- mse_loss=16705.95439453125 --- gen_adversarial_loss=1.8386092066764832
Step 79 --- TEST --- mse_loss=16676.029853144777
                  --- disc_adversarial_loss=1.400337745871725
                  --- mse_loss=16676.213558148735 --- gen_adversarial_loss=1.8373054220706602
```


