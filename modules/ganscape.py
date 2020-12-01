from keras.layers import Input, Activation, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Lambda
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras import losses, activations
from keras.callbacks import TensorBoard
import keras.backend as K
import tensorflow as tf

from network_structure import buildGenerator, buildDiscriminator
from custom_losses import generatorMSELossInJoint, generatorAdversarialLoss, discriminatorAdversarialLoss

class GANScape():
    def __init__(self, batchSize=32, inputImgShape=(64,64,3), localDiscInputShape=(28,28,3), generatorDescriber=None, discriminatorDescriber=None, optimizers=None, weightsForJointLoss=None):
        #self.imgHeight = 64
        #self.imgWidth = 64
        #self.channels = 4
        self.batchSize = batchSize
        self.inputImgShape = inputImgShape
        self.localDiscInputShape = localDiscInputShape
        self.weightForAdversarialInJointLoss = 0.1
        
        self.optimizers = optimizers
        self.weightsForJointLoss = weightsForJointLoss
        

        """
        #   One block = [Conv2D] + [ReLU] + [BatchNorm]
        #   parameters: filterNum, kernelSize, strideSize
        #   -> So on block
              Conv2D(filterNum, kernel_size=(kernelSize, kernelSize), strides=(strideSize, strideSize), padding='same')
              ReLU()
              BatchNormalization()
        """

        self.generatorDescriber = generatorDescriber
        self.discriminatorDescriber = discriminatorDescriber

        
    def compileModel(self, model, losses, optimizer, lossWeights=None):
        if lossWeights is None:
            model.compile(loss = losses,optimizer = optimizer)
        else:
            model.compile(loss = losses,optimizer = optimizer, loss_weights=lossWeights)
        return model

    #operation for lamda layer between [generator-output] -> [local-discriminator-input]
    def cropLocalDiscInputFromImg(self, realImgs, fakeImgs, maskParams):
        realImgs = tf.split(realImgs, self.batchSize)
        fakeImgs = tf.split(fakeImgs, self.batchSize)
        maskParams = tf.split(maskParams, self.batchSize)
        realLocalInputs = []
        fakeLocalInputs = []
        for real, fake, maskParam in zip(realImgs, fakeImgs, maskParams):
            fake = K.squeeze(fake, 0)
            real = K.squeeze(real, 0)
            maskParam = K.cast(K.squeeze(maskParam, 0), tf.int32)

            top = maskParam[0]
            left = maskParam[1]
            h = maskParam[2] - top + 1
            w = maskParam[3] - left + 1

            realImgLocalInput = tf.image.crop_to_bounding_box(
                real, top, left, h, w)
            realLocalInputs.append(realImgLocalInput)
            
            fakeImgLocalInput = tf.image.crop_to_bounding_box(
                fake, top, left, h, w)
            fakeLocalInputs.append(fakeImgLocalInput)


        realLocalInputs = K.stack(realLocalInputs)
        fakeLocalInputs = K.stack(fakeLocalInputs)

        return [realLocalInputs,  fakeLocalInputs]
        
    
    
    def buildAndCompile(self, generatorDescriber=None, discriminatorDescriber=None, optimizers=None, weightsForJointLoss=None, pathForLoadModel=None):
        #define inputs
        genInputShape = (self.inputImgShape[0], self.inputImgShape[1], self.inputImgShape[2])
        inputImgGen = Input(genInputShape, name='input_image_layer_for_generator')
        
        maskParamsInputLocalDisc = Input(shape=4, name='input_mask_params_for_local_disc')
        realInputLocalDisc = Input(shape=self.localDiscInputShape, name='input_real_mask_img_for_local_disc')
        
        realInputGlobalDisc = Input(shape=self.inputImgShape, name='input_real_img_for_global_disc')

        #handling loading model if needed
        if pathForLoadModel != None:
            self.generator = buildGenerator(inputImgGen, generatorDescriber)
            self.discriminator = buildDiscriminator(realInputGlobalDisc, realInputLocalDisc, discriminatorDescriber)

        else:
            self.generator = buildGenerator(inputImgGen, generatorDescriber)
            self.discriminator = buildDiscriminator(realInputGlobalDisc, realInputLocalDisc, discriminatorDescriber)


            
        fakeInputGlobalDisc = self.generator.layers[-1].output
    
        # local discriminator get the output of generator through a Lamda layer that cuts the inpainted part of the image
        realInputLocalDisc, fakeInputLocalDisc = Lambda(lambda x: self.cropLocalDiscInputFromImg(*x),name='crop_local_disc_input_from_img')([realInputGlobalDisc, fakeInputGlobalDisc, maskParamsInputLocalDisc])
        realProb =  self.discriminator([realInputGlobalDisc, realInputLocalDisc])
        fakeProb =  self.discriminator([fakeInputGlobalDisc, fakeInputLocalDisc])
        
        #print("prob_real: ", realProb)
        #print("prob_fake: ", fakeProb)
        def _stack(pReal, pFake):
            prob = K.squeeze(K.stack([pReal, pFake], -1), 1)
            #print(prob)
            return prob
        probs = Lambda(lambda x: _stack(*x), name='stack_prob')([realProb, fakeProb])
        # print("prob: ", prob)
        
        outputsForOnlyDisc = [probs]
        lossesForOnlyDisc = [discriminatorAdversarialLoss]
        outputsForOnlyGenJoint = [fakeInputGlobalDisc, probs]
        lossesForOnlyGenJoint = [generatorMSELossInJoint, generatorAdversarialLoss]
        lossWeightForOnlyGenJoint = [weightsForJointLoss["mse"], weightsForJointLoss["adversarial"]]
        
        ########################################################x
        #COMPILE TRAINING MODELS - we will have 3 different compiled variables (but the same network) becasuse we need:
        #   -different losses
        #   -different input-output combinations
        # in the 3 different phases of training
        
        combinedModelTrainOnlyDisc = Model([inputImgGen, maskParamsInputLocalDisc, realInputGlobalDisc], outputsForOnlyDisc)
        combinedModelTrainOnlyGenWithJoint = Model([inputImgGen, maskParamsInputLocalDisc, realInputGlobalDisc], outputsForOnlyGenJoint)
        
        self.modelForTrainOnlyGenWithMSE = self.compileModel(self.generator, 'mean_squared_error', optimizers["onlyGenMse"])
        self.modelForTrainOnlyDisc = self.compileModel(combinedModelTrainOnlyDisc, lossesForOnlyDisc, optimizers["onlyDisc"])
        self.modelForTrainOnlyGenWithJoint = self.compileModel(combinedModelTrainOnlyGenWithJoint, lossesForOnlyGenJoint, optimizers["onlyGenJoint"], lossWeights=lossWeightForOnlyGenJoint)
        
        #activatedGenOutput = Activation(activations.sigmoid)(self.generator.output)
        self.generatorForPredict = Model(inputImgGen, self.generator.output)

    