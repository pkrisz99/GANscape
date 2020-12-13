from PIL import Image
import PIL
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output
from datetime import datetime
import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
import time

from dataread import DataRead
from plotter import generateAndPlotImgs, plotImages


#function for training and validation
#   net: object of tpye GANScape - contains different compiled model variables (with the same network) for all of the 3 different train-phases
#   trainData: data container-handler for train data
#   validData: data container-handler for validation data
#   epochs: number of epoch
#   batchSize: size of batch during training and validation
#   stepsPerEpoch: numberOfImages // batchSize
#   ratios: describes the rate of the 3 train-cases
#   numOfBatchesLoadedAtOnce: number that decsribes how many batch of data we are storing in the memory at once


def trainNetwork(net, epochs, batchSize, stepsPerEpoch, ratios, numOfBatchesLoadedAtOnce, plotImgsDict, earlyStopPhases, whatToSave, paths):

    date_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    tensorboardLogWriter = tf.summary.create_file_writer(os.path.join(paths["logDirForTensorboard"],date_time) )
    tf.summary.experimental.set_step(step=1)

    tC = int(ratios["phaseOne"] * stepsPerEpoch["train"])
    tD = int(ratios["phaseTwo"] * stepsPerEpoch["train"])

    dummyLabels = np.empty((batchSize, 2))
    trainLoss = None
    validLoss = None

    bestValidLoss = {
        "genMse": 9999999999999999999,
        "genJoint": 9999999999999999999,
        "epochMse": "",
        "epochJoint": ""
    }

    stepForValidLosses = 1
    stepForTrainLosses = 0
    memoryForTrainLosses = {
        "phaseI": -1,
        "phaseII": -1,
        "phaseIII": {
            "disc":-1,
            "gen":[-1,-1,-1]
        }
    }

    net.discriminator.trainable = True
    net.generator.trainable = True

    stdOut = f"Training with \n  -epochs={epochs},\n  -batchSize={batchSize}\n  -steps for training={stepsPerEpoch['train']},\n  -steps for validation={stepsPerEpoch['valid']},\n  -phase I until step={tC},\n -phase II until step={tC+tD}\n\n"
    print(stdOut)

    with tensorboardLogWriter.as_default():
      for epoch in range(1, epochs+1):
          trainData = DataRead(paths["data"],'train', batchSize, numOfBatchesLoadedAtOnce, shuffle = True)
          validData = DataRead(paths["data"],'valid', batchSize, numOfBatchesLoadedAtOnce, shuffle = True)

          #------------------------------------------TRAINING-------------------------------------
          c = 0 #counter in actual loaded batches (for training)
          t = 1 #counter for steps in the epoch (for training)
          trainLoss = {
              "phaseI": {
                  "num": 0,
                  "loss": 0
              },
              "phaseII": {
                  "num": 0,
                  "loss": 0
              },
              "phaseIII":{
                  "num": 0,
                  "lossGen" : [0,0,0],
                  "lossDisc": 0
              }
          }
          while t <= stepsPerEpoch["train"]:
              if c == numOfBatchesLoadedAtOnce:
                  c = 0
                  trainData.reset()

              if t <= tC:
                  # ---------------------
                  #  PHASE ONE, train only Generatort with weighted MSE
                  # ---------------------
                  net.generator.trainable = True
                  net.discriminator.trainable = False
                  ret = net.modelForTrainOnlyGenWithMSE.train_on_batch(
                      x = trainData.cropped_images[c]/255,
                      y = trainData.target_images[c]/255,
                      reset_metrics=True)

                  #handle STDOUT logging for phase one
                  trainLoss["phaseI"]["num"] += 1
                  trainLoss["phaseI"]["loss"] += ret
                  actualInfo = f"---Epoch {epoch} --- Step {t} --- PHASE I --- gen_mse_loss={trainLoss['phaseI']['loss']/trainLoss['phaseI']['num']}"

                  #save phase one loss for TENSORBOARD logging
                  memoryForTrainLosses["phaseI"] = ret
                  tf.summary.scalar("train_I_gen_mse_loss", memoryForTrainLosses["phaseI"], step=stepForTrainLosses)





              elif t <= (tC + tD):
                  # ---------------------
                  #  PHASE TWO, train only Discriminator with binary crossentropy
                  # ---------------------
                  net.generator.trainable = False
                  net.discriminator.trainable = False
                  ret = net.modelForTrainOnlyDisc.train_on_batch(
                      x = [trainData.cropped_images[c]/255, trainData.csv[c,:,4:], trainData.target_images[c]/255],
                      y = dummyLabels,
                      reset_metrics=True)

                  #handle STDOUT logging for phase two
                  trainLoss["phaseII"]["num"] += 1
                  trainLoss["phaseII"]["loss"] += ret
                  discAdvCumSum = trainLoss['phaseII']['loss']/trainLoss['phaseII']['num']
                  actualInfo = f"---Epoch {epoch} --- Step {t} --- PHASE II --- disc_adversarial_loss={discAdvCumSum}"

                  #save phase two loss for TENSORBOARD logging
                  memoryForTrainLosses["phaseII"] = ret
                  tf.summary.scalar("train_II_disc_adversarial_loss", memoryForTrainLosses["phaseII"], step=stepForTrainLosses)

                  #"early stop" for phase II

                  if discAdvCumSum < earlyStopPhases["phaseTwo"]["minForDiscAdv"]:
                    t = tC + tD
                    actualInfo += " --- loss condition activated -> PHASE II ENDS NOW\n"


              else:
                  # ---------------------
                  #  PHASE THREE, train Generator with joint loss + Discriminator with binary crossentropy
                  # ---------------------
                  net.generator.trainable = True
                  net.discriminator.trainable = False
                  retG = net.modelForTrainOnlyGenWithJoint.train_on_batch(
                      x=[trainData.cropped_images[c]/255, trainData.csv[c,:,4:], trainData.target_images[c]/255],
                      y=[trainData.target_images[c]/255, dummyLabels],
                      reset_metrics=True)

                  net.generator.trainable = False
                  net.discriminator.trainable = True
                  retD = net.modelForTrainOnlyDisc.train_on_batch(
                      x = [trainData.cropped_images[c]/255, trainData.csv[c,:,4:], trainData.target_images[c]/255],
                      y = dummyLabels,
                      reset_metrics=True)

                  #handle STDOUT logging for phase three
                  trainLoss["phaseIII"]["num"] += 1
                  trainLoss["phaseIII"]["lossDisc"] += retD
                  trainLoss["phaseIII"]["lossGen"][0] += retG[0]
                  trainLoss["phaseIII"]["lossGen"][1] += retG[1]
                  trainLoss["phaseIII"]["lossGen"][2] += retG[2]
                  discAdvCumSum = trainLoss["phaseIII"]["lossDisc"]/trainLoss["phaseIII"]["num"]
                  genAdvCumSum = trainLoss["phaseIII"]["lossGen"][2]/trainLoss["phaseIII"]["num"]
                  actualInfo = f'---Epoch {epoch} --- Step {t} --- PHASE III --- disc_adversarial_loss={discAdvCumSum} --- gen_joint_loss={trainLoss["phaseIII"]["lossGen"][0]/trainLoss["phaseIII"]["num"]} --- gen_mse_loss={trainLoss["phaseIII"]["lossGen"][1]/trainLoss["phaseIII"]["num"]} --- gen_adversarial_loss={genAdvCumSum}'

                  #save phase three loss for TENSORBOARD logging
                  memoryForTrainLosses["phaseIII"]["disc"] = retD
                  memoryForTrainLosses["phaseIII"]["gen"][0] = retG[0]
                  memoryForTrainLosses["phaseIII"]["gen"][1] = retG[1]
                  memoryForTrainLosses["phaseIII"]["gen"][2] = retG[2]
                  tf.summary.scalar("train_III_disc_adversarial_loss", memoryForTrainLosses["phaseIII"]["disc"], step=stepForTrainLosses)
                  tf.summary.scalar("train_III_gen_joint_loss", memoryForTrainLosses["phaseIII"]["gen"][0], step=stepForTrainLosses)
                  tf.summary.scalar("train_III_gen_mse_loss", memoryForTrainLosses["phaseIII"]["gen"][1], step=stepForTrainLosses)
                  tf.summary.scalar("train_III_gen_adversarial_loss", memoryForTrainLosses["phaseIII"]["gen"][2], step=stepForTrainLosses)

                  #"early stop" for phase III
                  if (discAdvCumSum < earlyStopPhases["phaseThree"]["minForDiscAdv"]) or (genAdvCumSum > earlyStopPhases["phaseThree"]["maxForGenAdv"]):
                    t = stepsPerEpoch["train"]
                    actualInfo += " --- loss condition activated -> PHASE III ENDS NOW\n"



              clear_output()
              print(stdOut + actualInfo)
              #save std out if phase-switch will happen in next iteration or this is last iteration in epoch
              if (t == tC) or (t == tC + tD) or (t == stepsPerEpoch["train"]):
                  stdOut += actualInfo + "\n"


              #save model if configured period implies that or this is last iteration in epoch
              """
              if (t % periodForSaveModel == 0) or (t == stepsPerEpoch["train"]):
                stdOut += actualInfo + ' --- model saved after this step ---' + "\n"
                pass
              """

              #handle TENSORBOARD logging
              """
              if memoryForTrainLosses["phaseI"] >= 0:
                tf.summary.scalar("gen_mse_loss", memoryForTrainLosses["phaseI"], step=stepForTrainLosses)
              if memoryForTrainLosses["phaseII"] >= 0:
                tf.summary.scalar("disc_adversarial_loss", memoryForTrainLosses["phaseII"], step=stepForTrainLosses)
              if min(memoryForTrainLosses["phaseIII"]) >= 0:
                tf.summary.scalar("gen_joint_loss", memoryForTrainLosses["phaseIII"][0], step=stepForTrainLosses)
                tf.summary.scalar("gen_mse_loss_in_joint", memoryForTrainLosses["phaseIII"][1], step=stepForTrainLosses)
                tf.summary.scalar("gen_adversarial_loss_in_joint", memoryForTrainLosses["phaseIII"][2], step=stepForTrainLosses)
              """
              tensorboardLogWriter.flush()

              #update iteration counters
              t += 1
              c += 1
              stepForTrainLosses += 1

          #------------------------------------------------VALIDATION-------------------------------------
          c = 0 #counter in actual loaded batches (for validation)
          t = 0 #counter for steps in the epoch (for validation)
          validLoss = {
              "num": 0,
              "genOnlyMSE": 0,
              "discAdversarial": 0,
              "genJoint": [0,0,0]
              }
          while t < stepsPerEpoch["valid"]:
              if c == numOfBatchesLoadedAtOnce:
                  c = 0
                  validData.reset()

              actualInfo =""
              validLoss["num"] += 1
              """
              retGenMSE = net.modelForTrainOnlyGenWithMSE.test_on_batch(
                          x = validData.cropped_images[c],
                          y = validData.target_images[c],
                          reset_metrics=True)
              validLoss["genOnlyMSE"] += retGenMSE
              actualInfo += f"---Epoch {epoch} --- Step {t+1} --- VALIDATION --- mse_loss={validLoss['genOnlyMSE']/validLoss['num']}"
              """

              retD = net.modelForTrainOnlyDisc.test_on_batch(
                          x = [validData.cropped_images[c]/255, validData.csv[c,:,4:], validData.target_images[c]/255],
                          y = dummyLabels,
                          reset_metrics=True)

              validLoss["discAdversarial"] += retD

              actualInfo += "\n"
              actualInfo += f"---Epoch {epoch} --- Step {t+1} --- VALIDATION --- disc_adversarial_loss={validLoss['discAdversarial']/validLoss['num']}"

              retGJoint = net.modelForTrainOnlyGenWithJoint.test_on_batch(
                          x = [validData.cropped_images[c]/255, validData.csv[c,:,4:], validData.target_images[c]/255],
                          y = [validData.target_images[c]/255, dummyLabels],
                          sample_weight=None,
                          reset_metrics=True)

              validLoss["genJoint"][0] += retGJoint[0]
              validLoss["genJoint"][1] += retGJoint[1]
              validLoss["genJoint"][2] += retGJoint[2]
              actualInfo += "\n"
              actualInfo += f"---Epoch {epoch} --- Step {t+1} --- VALIDATION --- gen_joint_loss={validLoss['genJoint'][0]/validLoss['num']} --- gen_mse_loss={validLoss['genJoint'][1]/validLoss['num']} --- gen_adversarial_loss={validLoss['genJoint'][2]/validLoss['num']}"

              clear_output()
              print(stdOut + actualInfo)
              #save std out if this is the last iteration in epoch
              if (t == stepsPerEpoch["valid"]):
                  stdOut += actualInfo + "\n"


              #tf.summary.scalar("valid_I_gen_mse_loss", retGenMSE, step=stepForValidLosses)
              tf.summary.scalar("valid_disc_adversarial_loss", retD, step=stepForValidLosses)
              tf.summary.scalar("valid_gen_joint_loss", retGJoint[0], step=stepForValidLosses)
              tf.summary.scalar("valid_gen_mse_loss", retGJoint[1], step=stepForValidLosses)
              tf.summary.scalar("valid_gen_adversarial_loss", retGJoint[2], step=stepForValidLosses)
              tensorboardLogWriter.flush()

              t += 1
              c += 1
              stepForValidLosses += 1


          #------------------------------------------------SAVE MODEL--------------------------------------
          dateFolder = os.path.join(paths["saveModelDir"], date_time)

          if whatToSave == 'best':
              finalGenMseLoss = validLoss['genJoint'][1]/validLoss['num']
              if bestValidLoss["genMse"] > finalGenMseLoss:
                net.generator.save(os.path.join(os.path.join(dateFolder, "gen_mse_best"), "generator"))
                net.discriminator.save(os.path.join(os.path.join(dateFolder, "gen_mse_best"), "discriminator"))
                bestValidLoss["genMse"] = finalGenMseLoss
                bestValidLoss["epochMse"] = epoch

              finalGenJointLoss = validLoss['genJoint'][0]/validLoss['num']
              if bestValidLoss["genJoint"] > finalGenJointLoss:
                net.generator.save(os.path.join(os.path.join(dateFolder, "gen_joint_best"), "generator"))
                net.discriminator.save(os.path.join(os.path.join(dateFolder, "gen_joint_best"), "discriminator"))
                bestValidLoss["genJoint"] = finalGenJointLoss
                bestValidLoss["epochJoint"] = epoch

              f= open(os.path.join(dateFolder, "info.txt"),"w+")
              f.write(f'Saved best model (according to gen_(join_mse loss): epoch = {bestValidLoss["epochMse"]}\n')
              f.write(f'Saved best model (according to generator joint loss) : epoch = {bestValidLoss["epochJoint"]}\n')
              f.close()

          elif whatToSave == 'all':
              net.generator.save(os.path.join(os.path.join(dateFolder, f'epoch_{epoch}'), "generator"))
              net.discriminator.save(os.path.join(os.path.join(dateFolder, f'epoch_{epoch}'), "discriminator"))

          #------------------------------------------------SHOW IMAGES-------------------------------------
          if plotImgsDict["isUsed"]:
            if plotImgsDict["saveImgsToFile"]:
              imgMaps = generateAndPlotImgs(net, validData, plotImgsDict, numOfBatchesLoadedAtOnce, True)
              dirPath =  os.path.join(os.path.join(paths["saveImgsDirDuringTrain"], date_time), f'epoch_{epoch}')
              if not os.path.exists(dirPath):
                os.makedirs(dirPath)
              """
              for ind in range(plotImgsDict["numOfImgs"]):
                Image.fromarray(imgsToSave["targetWhole"][ind].astype('uint8')).save(dirPath+f'/{ind}_target_whole.jpg')
                Image.fromarray(imgsToSave["targetCroppedWhole"][ind].astype('uint8')).save(dirPath+f'/{ind}_target_cropped_whole.jpg')
                Image.fromarray(imgsToSave["generatedWhole"][ind].astype('uint8')).save(dirPath+f'/{ind}_generated_whole.jpg')
                Image.fromarray(imgsToSave["targetCropPart"][ind].astype('uint8')).save(dirPath+f'/{ind}_target_crop_part.jpg')
                Image.fromarray(imgsToSave["generatedCropPart"][ind].astype('uint8')).save(dirPath+f'/{ind}_generated_crop_part.jpg')
              """
              [Image.fromarray(map.astype('uint8')).save(dirPath+f'/{i}_IMG_MAP.jpg') for i, map in enumerate(imgMaps)]
            else:
              generateAndPlotImgs(net, testData, plotImgsDict, numOfBatchesLoadedAtOnce, False)

            plt.show()
            time.sleep(plotImgsDict["sleepingTime"])
    return trainLoss, validLoss


#-----------------------------------------------------------------------------------------------------------------------------

#function for test
#   net: object of tpye GANScape
#   testData: data container-handler for test data
#   validData: data container-handler for validation data
#   batchSize: size of batch during training and validation
#   stepsPerEpochDuringTest: numberOfImages // batchSize
#   numOfBatchesLoadedAtOnce: number that decsribes how many batch of data we are storing in the memory at once

def testNetwork(net, batchSize, stepsForTest, numOfBatchesLoadedAtOnce, plotImgsDict, paths):
      testData = DataRead(paths["data"], 'test', batchSize, numOfBatchesLoadedAtOnce, shuffle = True)
      date_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
      c = 0 #counter in actual loaded batches (for validation)
      t = 1 #counter for steps in the epoch (for validation)
      testLoss = {
          "num":0,
          "genOnlyMSE": 0,
          "discAdversarial": 0,
          "genJoint": [0,0,0]
          }
      dummyLabels = np.empty((batchSize, 2))
      stdOut = f"Test with \n  -batchSize={batchSize}\n  -steps={stepsForTest}\n\n"
      while t <= stepsForTest:
          if c == numOfBatchesLoadedAtOnce:
              c = 0
              testData.reset()

          actualInfo =""
          testLoss["num"] += 1

          retGenMSE = net.modelForTrainOnlyGenWithMSE.test_on_batch(
                      x = testData.cropped_images[c]/255,
                      y = testData.target_images[c]/255,
                      reset_metrics=True)
          testLoss["genOnlyMSE"] += retGenMSE
          actualInfo += f"Step {t} --- TEST --- mse_loss={testLoss['genOnlyMSE']/testLoss['num']}"

          retD = net.modelForTrainOnlyDisc.test_on_batch(
                      x = [testData.cropped_images[c]/255, testData.csv[c,:,4:], testData.target_images[c]/255],
                      y = dummyLabels,
                      sample_weight=None,
                      reset_metrics=True)

          testLoss["discAdversarial"] += retD
          actualInfo += "\n"
          actualInfo += f"Step {t}        --- TEST --- disc_adversarial_loss={testLoss['discAdversarial']/testLoss['num']}"

          retGJoint = net.modelForTrainOnlyGenWithJoint.test_on_batch(
                      x = [testData.cropped_images[c]/255, testData.csv[c,:,4:], testData.target_images[c]/255],
                      y = [testData.target_images[c]/255, dummyLabels],
                      sample_weight=None,
                      reset_metrics=True)

          testLoss["genJoint"][0] += retGJoint[0]
          testLoss["genJoint"][1] += retGJoint[1]
          testLoss["genJoint"][2] += retGJoint[2]

          actualInfo += "\n"
          actualInfo += f"                 --- TEST --- gen_joint_loss={testLoss['genJoint'][0]/testLoss['num']}--- gen_mse_loss={testLoss['genJoint'][1]/testLoss['num']} --- gen_adversarial_loss={testLoss['genJoint'][2]/testLoss['num']}"

          clear_output()
          print(stdOut + actualInfo)
          t += 1
          c += 1

      if plotImgsDict["isUsed"]:
        if plotImgsDict["saveImgsToFile"]:
          imgMaps = generateAndPlotImgs(net, testData, plotImgsDict, numOfBatchesLoadedAtOnce, True)
          dirPath =  os.path.join(os.path.join(paths["saveImgsDirDuringTest"], date_time))
          if not os.path.exists(dirPath):
            os.makedirs(dirPath)
          """
          for ind in range(plotImgsDict["numOfImgs"]):
            Image.fromarray(imgsToSave["targetWhole"][ind].astype('uint8')).save(dirPath+f'/target_whole.jpg')
            Image.fromarray(imgsToSave["targetCroppedWhole"][ind].astype('uint8')).save(dirPath+f'/target_cropped_whole.jpg')
            Image.fromarray(imgsToSave["generatedWhole"][ind].astype('uint8')).save(dirPath+f'/generated_whole.jpg')
            Image.fromarray(imgsToSave["targetCropPart"][ind].astype('uint8')).save(dirPath+f'/target_crop_part.jpg')
            Image.fromarray(imgsToSave["generatedCropPart"][ind].astype('uint8')).save(dirPath+f'/generated_crop_part.jpg')
          """
          [Image.fromarray(map.astype('uint8')).save(dirPath+f'/{i}_IMG_MAP.jpg') for i, map in enumerate(imgMaps)]
        else:
          generateAndPlotImgs(net, testData, plotImgsDict, numOfBatchesLoadedAtOnce, False)



      return testLoss