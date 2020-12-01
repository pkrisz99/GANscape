from keras.layers import Input, Activation, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Lambda
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras import losses, activations

def buildNormalConvolutionalBlock(input, filterNum, kernelSize, strideSize, dilationSize):
        hidden = Conv2D(filterNum, kernel_size=(kernelSize, kernelSize), strides=(strideSize, strideSize), dilation_rate=(dilationSize, dilationSize) , padding='same')(input)
        hidden = ReLU()(hidden)
        out = BatchNormalization()(hidden)
        return out
    
def buildTransposedConvolutionalBlock(input, filterNum, kernelSize, strideSize):
    hidden = Conv2DTranspose(filterNum, kernel_size=(kernelSize, kernelSize), strides=(strideSize, strideSize), padding='same')(input)
    hidden = ReLU()(hidden)
    out = BatchNormalization()(hidden)
    return out

def buildGenerator(inputImageLayer, generatorDescriber):
        gen = inputImageLayer
        
        #building EARLY PART of generator
        for i in range(generatorDescriber["earlyPart"]["numOfBlocks"]):
          gen = buildNormalConvolutionalBlock(gen, 
                                                   generatorDescriber["earlyPart"]["filterNums"][i], 
                                                   generatorDescriber["earlyPart"]["kernelSizes"][i], 
                                                   generatorDescriber["earlyPart"]["strideSizes"][i], 
                                                   1)

        #building MID PART of generator
        for i in range(generatorDescriber["midPart"]["numOfBlocksWithDilation"]):
          actDilationRate = pow(2,i)
          gen = buildNormalConvolutionalBlock(gen, 
                                                   generatorDescriber["midPart"]["filterNum"], 
                                                   3, 
                                                   1, 
                                                   actDilationRate)

        #building END PART of generator
        for i in range(generatorDescriber["endPart"]["numOfBlocks"]):
          if (generatorDescriber["endPart"]["convolutionType"][i] == "normal"):
            gen = buildNormalConvolutionalBlock(gen, 
                                                     generatorDescriber["endPart"]["filterNums"][i], 
                                                     generatorDescriber["endPart"]["kernelSizes"][i], 
                                                     generatorDescriber["endPart"]["strideSizes"][i], 
                                                     1)
          elif (generatorDescriber["endPart"]["convolutionType"][i] == "transposed"):
            gen = buildTransposedConvolutionalBlock(gen, 
                                                         generatorDescriber["endPart"]["filterNums"][i], 
                                                         generatorDescriber["endPart"]["kernelSizes"][i], 
                                                         generatorDescriber["endPart"]["strideSizes"][i])
        
        #build output 
        genOutput = Conv2D(3, kernel_size=(3, 3), padding='same')(gen)
        genOutput = Activation(activations.sigmoid)(genOutput)
        
        #return Model([inputImageLayer, inputMaskLayer], genOutput)
        return Model(inputImageLayer, genOutput)


def buildDiscriminator(globalDiscInput, localDiscInput, discriminatorDescriber):
        globalDisc = globalDiscInput
        #building GLOBAL discriminator
        for i in range(discriminatorDescriber["global"]["numOfBlocks"]):
          globalDisc = buildNormalConvolutionalBlock(globalDisc, 
                                                          discriminatorDescriber["global"]["filterNums"][i], 
                                                          discriminatorDescriber["global"]["kernelSizes"][i], 
                                                          discriminatorDescriber["global"]["strideSizes"][i], 
                                                          1)
        globalDisc = Conv2D(512, kernel_size=(3, 3), padding='same')(globalDisc)
        if discriminatorDescriber["useGlobalAvgPool"]:
          globalDisc = GlobalAveragePooling2D()(globalDisc)
        else:
          globalDisc = ReLU()(globalDisc)
          globalDisc = Flatten()(globalDisc)
          globalDisc = Dense(512)(globalDisc)
        globalDisc = Dense(256)(globalDisc)
          

        localDisc = localDiscInput
        #building LOCAL discriminator
        for i in range(discriminatorDescriber["local"]["numOfBlocks"]):
          localDisc = buildNormalConvolutionalBlock(localDisc, 
                                                         discriminatorDescriber["local"]["filterNums"][i], 
                                                         discriminatorDescriber["local"]["kernelSizes"][i], 
                                                         discriminatorDescriber["local"]["strideSizes"][i], 
                                                         1)
        localDisc = Conv2D(512, kernel_size=(5, 5), strides=(2,2), padding='same')(localDisc)
        if discriminatorDescriber["useGlobalAvgPool"]:
          localDisc = GlobalAveragePooling2D()(localDisc)
        else:
          localDisc = ReLU()(localDisc)
          localDisc = Flatten()(localDisc)
          localDisc = Dense(512)(localDisc)
        localDisc = Dense(256)(localDisc)

        
        concatedDisc = Concatenate()([globalDisc, localDisc])
        concatedDisc = Dense(1)(concatedDisc)
        
        return Model(inputs=[globalDiscInput, localDiscInput], outputs=concatedDisc)