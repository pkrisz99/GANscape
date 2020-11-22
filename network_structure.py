from keras.layers import Input, Activation, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Lambda
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras import losses, activations


def buildGenerator(inputImageLayer):
    gen = Conv2D(64, kernel_size=(5, 5), padding='same')(inputImageLayer)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    
    gen = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    gen = Conv2D(128, kernel_size=(3, 3), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    
    gen = Conv2D(256, kernel_size=(3, 3), strides=(2,2), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    gen = Conv2D(256, kernel_size=(3, 3), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    gen = Conv2D(256, kernel_size=(3, 3), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    
    gen = Conv2D(256, kernel_size=(3, 3), dilation_rate=(2, 2), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    gen = Conv2D(256, kernel_size=(3, 3), dilation_rate=(4, 4), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    gen = Conv2D(256, kernel_size=(3, 3), dilation_rate=(8, 8), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    gen = Conv2D(256, kernel_size=(3, 3), dilation_rate=(16, 16), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    
    gen = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    gen = Conv2D(128, kernel_size=(3, 3), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    
    gen = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    gen = Conv2D(32, kernel_size=(3, 3), padding='same')(gen)
    gen = ReLU()(gen)
    gen = BatchNormalization()(gen)
    
    genOutput = Conv2D(3, kernel_size=(3, 3), padding='same')(gen)
    #genOutput = Activation(activations.sigmoid)(genOutput)
    
    #return Model([inputImageLayer, inputMaskLayer], genOutput)
    return Model(inputImageLayer, genOutput)
    
    
def buildDiscriminator(globalDiscInput, localDiscInput):
    globalDisc = Conv2D(64, kernel_size=(5, 5), strides=(2,2), padding='same')(globalDiscInput)
    globalDisc = ReLU()(globalDisc)
    globalDisc = BatchNormalization()(globalDisc)
    globalDisc = Conv2D(128, kernel_size=(5, 5), strides=(2,2), padding='same')(globalDisc)
    globalDisc = ReLU()(globalDisc)
    globalDisc = BatchNormalization()(globalDisc)
    globalDisc = Conv2D(256, kernel_size=(5, 5), strides=(2,2), padding='same')(globalDisc)
    globalDisc = ReLU()(globalDisc)
    globalDisc = BatchNormalization()(globalDisc)
    globalDisc = Conv2D(512, kernel_size=(5, 5), strides=(2,2), padding='same')(globalDisc)
    globalDisc = ReLU()(globalDisc)
    globalDisc = BatchNormalization()(globalDisc)
    globalDisc = Conv2D(512, kernel_size=(5, 5), strides=(2,2), padding='same')(globalDisc)
    globalDisc = ReLU()(globalDisc)
    globalDisc = BatchNormalization()(globalDisc)
    globalDisc = Conv2D(512, kernel_size=(5, 5), strides=(2,2), padding='same')(globalDisc)
    globalDisc = ReLU()(globalDisc)

    globalDisc = Flatten()(globalDisc)
    globalDisc = Dense(1024)(globalDisc)
    
    
    
    localDisc = Conv2D(64, kernel_size=(5, 5), strides=(2,2), padding='same')(localDiscInput)
    localDisc = ReLU()(localDisc)
    localDisc = BatchNormalization()(localDisc)
    localDisc = Conv2D(128, kernel_size=(5, 5), strides=(2,2), padding='same')(localDisc)
    localDisc = ReLU()(localDisc)
    localDisc = BatchNormalization()(localDisc)
    localDisc = Conv2D(256, kernel_size=(5, 5), strides=(2,2), padding='same')(localDisc)
    localDisc = ReLU()(localDisc)
    localDisc = BatchNormalization()(localDisc)
    localDisc = Conv2D(512, kernel_size=(5, 5), strides=(2,2), padding='same')(localDisc)
    localDisc = ReLU()(localDisc)
    localDisc = BatchNormalization()(localDisc)
    localDisc = Conv2D(512, kernel_size=(5, 5), strides=(2,2), padding='same')(localDisc)
    localDisc = ReLU()(localDisc)
    
    localDisc = Flatten()(localDisc)
    localDisc = Dense(1024)(localDisc)
    
    
    concatedDisc = Concatenate()([globalDisc, localDisc])
    concatedDisc = Dense(1)(concatedDisc)
    genOutput = Activation(activations.sigmoid)(concatedDisc)
    
    return Model(inputs=[globalDiscInput, localDiscInput], outputs=concatedDisc)
