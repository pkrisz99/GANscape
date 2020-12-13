import tensorflow as tf
import keras.backend as K


#mse loss for the generator in phase III
def generatorMSELossInJoint(y_true, y_pred):
    #print("weightedMSELoss --- y_true.shape=",y_true.shape, "y_pred.shape=",y_pred.shape)
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)


#adversarial loss for the generator in phase III
def generatorAdversarialLoss(y_true, y_pred):
        #print("generatorAdversarialLoss --- y_true.shape=", y_true.shape, "y_pred.shape=", y_pred.shape)
        real = y_pred[:, 0]
        fake = y_pred[:, 1]
        loss_real = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.zeros_like(real)))
        loss_fake = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))
        loss = loss_real + loss_fake
        return loss

#adversarial loss for the discriminator in phase II
def discriminatorAdversarialLoss(y_true, y_pred):
        real = y_pred[:, 0]
        fake = y_pred[:, 1]
        loss_real = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        loss_fake = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        loss = loss_real + loss_fake
        # loss = util.tfprint(loss, "discriminator_loss")
        return loss
