import numpy as np
import matplotlib.pyplot as plt


def plotImages(numOfImgs, figSize, globalRealImgs, globalCroppedImages, globalFakeImgs, localRealImgs, localFakeImgs, returnImgMap=False):
  imgMap = np.ones((5*80, numOfImgs*80, 3)) * 255
  fig, axes = plt.subplots(5, numOfImgs, figsize=figSize)
  rowLabels = ['target img without mask',
               'target img with mask',
               'generated img mask',
               'cropped part of target',
               'cropped part of generated']
  for i in range(0, numOfImgs):
      if i == 0:
        for j, label in enumerate(rowLabels):
          axes[j,i].set_ylabel(label, rotation=0, size='large')

      # real, target img without mask (64x64)
      #ax = plt.subplot(5, numOfImgs, i)
      axes[0,i].imshow((globalRealImgs[i]).astype('uint8'))
      axes[0,i].get_xaxis().set_visible(False)
      axes[0,i].get_yaxis().set_visible(False)
      imgMap[:64, i*80:(i*80)+64, :] = globalRealImgs[i]

      # real, target img with mask (64x64)
      #ax = plt.subplot(5, numOfImgs, i + numOfImgs)
      axes[1,i].imshow((globalCroppedImages[i]).astype('uint8'))
      axes[1,i].get_xaxis().set_visible(False)
      axes[1,i].get_yaxis().set_visible(False)
      imgMap[80:(80+64), i*80:(i*80)+64, :] = globalCroppedImages[i]

      # generated/fake, whole img (generator output) (64x64)
      #ax = plt.subplot(5, numOfImgs, i + 2*numOfImgs)
      axes[2,i].imshow((globalFakeImgs[i]).astype('uint8'))
      axes[2,i].get_xaxis().set_visible(False)
      axes[2,i].get_yaxis().set_visible(False)
      imgMap[2*80:(2*80)+64, i*80:(i*80)+64, :] = globalFakeImgs[i]

      # real crop img (28x28)
      #ax = plt.subplot(5, numOfImgs, i + 3*numOfImgs)
      axes[3,i].imshow((localRealImgs[i]).astype('uint8'))
      axes[3,i].get_xaxis().set_visible(False)
      axes[3,i].get_yaxis().set_visible(False)
      imgMap[3*80:(3*80)+28, (i*80+18):(i*80+18)+28, :] = localRealImgs[i]
      
      # generated crop img (28x28)
      #ax = plt.subplot(5, numOfImgs, i + 4*numOfImgs)
      axes[4,i].imshow((localFakeImgs[i]).astype('uint8'))
      axes[4,i].get_xaxis().set_visible(False)
      axes[4,i].get_yaxis().set_visible(False)
      imgMap[4*80:(4*80)+28, (i*80+18):(i*80+18)+28, :] = localFakeImgs[i]

  
  #for ax, label in zip(axes[:,0], rowLabels):
    #ax.set_ylabel(label, rotation=0, size='large')


  if returnImgMap:
    return imgMap
  else:
    return None



def generateAndPlotImgs(net, dataObj, plotImgsDict, numOfBatchesLoadedAtOnce, returnImgsToSave=False):
    assert plotImgsDict["numOfImgs"] <= dataObj.target_images.shape[1], 'plotImgsDit["numOfImgs"] must be smaller or equal than batchSize!'
    
    imgsObjs = []
    imgMaps = []

    batchInd = int(np.random.uniform(low=0, high=numOfBatchesLoadedAtOnce-0.00001))
    for i in range(plotImgsDict["numOfImgMapsToSave"]):
      start = i * plotImgsDict["numOfImgs"]
      end = (i+1) * plotImgsDict["numOfImgs"]
      globalFakeImgs = net.generatorForPredict.predict(
                    x=dataObj.cropped_images[batchInd, start:end]/255,
                    batch_size=plotImgsDict["numOfImgs"]
                )

      localFakeImgs = [actImg[params[4]:params[6]+1, params[5]:params[7]+1, :] for actImg, params in zip(globalFakeImgs, dataObj.csv[batchInd, start:end])]
      #j = 0
      #for actImg, params in zip(globalFakeImgs, dataObj.csv[batchInd, start:end]):
      #  print(f"{i}th imgmap, {j}th img, csv params={params}\n")
      localFakeImgs = np.asarray(localFakeImgs)

      for actImg, params in zip(globalFakeImgs, dataObj.csv[batchInd, start:end]):
        actImg[params[4]:params[6]+1, params[5]:params[7]+1, :] = 0

      #scale output values from [0, 1] to [0, 255]
      globalFakeImgs = (globalFakeImgs*255).astype('uint8')
      localFakeImgs = (localFakeImgs*255).astype('uint8')

      if returnImgsToSave:
        imgMap = plotImages(plotImgsDict["numOfImgs"],
                plotImgsDict["figSize"],
                dataObj.target_images[batchInd, start:end],
                dataObj.cropped_images[batchInd, start:end],
                globalFakeImgs,
                dataObj.crop_images[batchInd, start:end],
                localFakeImgs,
                True)
        """
        imgsObj = {
            "targetWhole": dataObj.target_images[batchInd, :plotImgsDict["numOfImgs"]+1],
            "targetCroppedWhole": dataObj.cropped_images[batchInd, :plotImgsDict["numOfImgs"]+1],
            "generatedWhole": globalFakeImgs,
            "targetCropPart": dataObj.crop_images[batchInd, :plotImgsDict["numOfImgs"]+1],
            "generatedCropPart": localFakeImgs
        }
        """
        imgMaps.append(imgMap)
      else:
        plotImages(plotImgsDict["numOfImgs"],
                plotImgsDict["figSize"],
                dataObj.target_images[batchInd, start:end],
                dataObj.cropped_images[batchInd, start:end],
                globalFakeImgs,
                dataObj.crop_images[batchInd, start:end],
                localFakeImgs,
                False)
        
    if returnImgsToSave:
      return imgMaps
    else:
      return None

