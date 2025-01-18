'''
    Implements loss functions to train the neural network.
    The loss functions combine Dice Coefficient Loss and Binary Cross-Entropy Loss
    (Common in segmentation tasks)
    BCE is good for pixel-wise accuracy
    Dice Loss improves region overlap
    By combining the two, the model gets the benefits of both: pixel-level accuracy from BCE and region-level accuracy
    from Dice Loss.
'''

import tensorflow as tf

# Dice Coefficient is a measure of how well the predicted mask matches the ground truth
def diceCoef(y_true, y_pred, smooth = tf.keras.backend.epsilon()):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f*y_pred_f)
    return (2. * intersection + smooth)/(tf.keras.backend.sum(y_true_f*y_true_f) + tf.keras.backend.sum(y_pred_f*y_pred_f) + smooth)

# Calculates the loss based on Dice Coefficient (lower is better)
def diceCoefloss(y_true, y_pred):
    return 1.0 - diceCoef(y_true,y_pred)

# Combines BCE and Dice Loss to create a hybrid loss function for training the model
def bceDiceLoss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + diceCoefloss(y_true, y_pred)
    return loss
