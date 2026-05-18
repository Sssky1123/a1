import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
# import paddle
import tensorflow as tf

print('GPU',tf.test.is_gpu_available())

import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import *
from tensorflow.contrib.keras import models, layers, regularizers
from tensorflow.contrib.keras import backend as K
import pandas as pd
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as compare_ssim
# from scipy.io import loadmat
from scipy import io

nBlockT = 32
nBlockX = 32
input_num = 1
output_num = 1



def myloss(y_true, y_pred):

    l1  = K.mean(K.abs(y_true - y_pred), axis=-1)
    loss = l1

    return loss

class myMDnet(object):

    def __init__(self, img_rows=None, img_cols=None):
        self.img_rows = img_rows
        self.img_cols = img_cols


    def unet(self,m=[]):
        if input_num == 1:
            input_m = m
        else:
            input_m = concatenate(m, axis=3)
        nm = 64
        kn = 3
        ag = 1e-5
        conv1 = Conv2D(nm, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(input_m)
        # conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Conv2D(nm, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(conv1)
        # conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

        conv2 = Conv2D(nm * 2, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(pool1)
        # conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Conv2D(nm * 2, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(conv2)
        # conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

        conv3 = Conv2D(nm * 4, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(pool2)
        # conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv2D(nm * 4, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(conv3)
        # conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

        conv4 = Conv2D(nm * 8, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(pool3)
        # conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Conv2D(nm * 8, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(conv4)
        # conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        up5 = Conv2DTranspose(nm * 4, (2, 2), strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(ag))(conv4)
        print(up5.shape)
        merge5 = concatenate([conv3, up5], axis=3)
        print(merge5.shape)

        conv5 = Conv2D(nm * 4, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(merge5)
        # conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = Conv2D(nm * 4, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(conv5)
        # conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        up6 = Conv2DTranspose(nm * 2, (2, 2), strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(ag))(conv5)
        merge6 = concatenate([conv2, up6], axis=3)

        conv6 = Conv2D(nm * 2, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(merge6)
        # conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        conv6 = Conv2D(nm * 2, kn, padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(conv6)
        # conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        up7 = Conv2DTranspose(nm, (2, 2), strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(ag))(conv6)
        merge7 = concatenate([conv1, up7], axis=3)

        conv7 = Conv2D(nm, kn, activation='relu', padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(merge7)
        conv7 = Conv2D(nm, kn, activation='relu', padding='same', kernel_initializer='glorot_normal',
                       kernel_regularizer=regularizers.l2(ag))(conv7)
        conv7 = Conv2D(output_num, 1, activation='tanh')(conv7)

        return conv7



    def get_MDnet(self):
        if input_num == 1:
            m = Input((nBlockT, nBlockX, 1))
            y = self.unet(m)
        else:
            m=list()
            for i in range(input_num):
                m1 = Input((nBlockT, nBlockX, 1))
                m.append(m1)
            y = self.unet(m)


        out = list()
        for i in range(output_num):
            output1 = Reshape((nBlockT, nBlockX, 1))(y[:, :, :, i])
            out.append(output1)
        model = Model(inputs=m, outputs=out)


        return model



    def matblend(self, orgg, nGather, nTrace, nPoint, nBlockT, nBlockX, novv):
        nOverlapT = math.floor(nBlockT / novv)
        nOverlapX = math.floor(nBlockX / novv)

        weightX = np.hanning(nBlockT + 2)[1:-1]
        weightY = np.hanning(nBlockX + 2)[1:-1]
        Weight2D = np.kron(weightX, np.transpose(weightY))
        Weight2D = np.reshape(Weight2D, (nBlockT, nBlockX))

        nBx = math.ceil((nPoint - nOverlapT) / (nBlockT - nOverlapT))
        if nBlockT == nPoint:
            nBx = 1
        nBy = math.ceil((nTrace - nOverlapX) / (nBlockX - nOverlapX))
        if nBlockX == nTrace:
            nBy = 1

        ePriL2 = np.zeros((nPoint, nGather * nTrace))
        ePriWeightL2 = np.zeros((nPoint, nGather * nTrace))

        for ii in range(nGather):

            for iBx in range(nBx):
                xStart = iBx * (nBlockT - nOverlapT) + 1
                xEnd = xStart + nBlockT - 1
                if xEnd > nPoint:
                    xEnd = nPoint
                    xStart = xEnd - nBlockT + 1
                for iBy in range(nBy):
                    yStart = iBy * (nBlockX - nOverlapX) + 1
                    yEnd = yStart + nBlockX - 1
                    if yEnd > nTrace:
                        yEnd = nTrace
                        yStart = yEnd - nBlockX + 1

                    orgBlock = orgg[ii * nBx * nBy + iBx * nBy + iBy, :, :]
                    orgvv = np.reshape(orgBlock, (nBlockT, nBlockX), 'F')

                    ePriL2[xStart - 1:xEnd, ii * nTrace + yStart - 1:ii * nTrace + yEnd] = ePriL2[xStart - 1:xEnd,
                                                                                           ii * nTrace + yStart - 1:ii * nTrace + yEnd] + orgvv * Weight2D
                    ePriWeightL2[xStart - 1:xEnd, ii * nTrace + yStart - 1:ii * nTrace + yEnd] = ePriWeightL2[
                                                                                                 xStart - 1:xEnd,
                                                                                                 ii * nTrace + yStart - 1: ii * nTrace + yEnd] + Weight2D

        for ii in range(nPoint):
            for jj in range(nGather * nTrace):
                if ePriWeightL2[ii, jj] == 0:
                    ePriWeightL2[ii, jj] = 1
                ePriL2[ii, jj] = ePriL2[ii, jj] / ePriWeightL2[ii, jj]

        return ePriL2

    def window_strate_vec(self, orgg, nBlockT, nBlockX, nTrace, nPoint, nGather, novv):
        nOverlapT = math.floor(nBlockT / novv)
        nOverlapX = math.floor(nBlockX / novv)
        nBx = math.ceil((nPoint - nOverlapT) / (nBlockT - nOverlapT))
        if nBlockT == nPoint:
            nBx = 1
        nBy = math.ceil((nTrace - nOverlapX) / (nBlockX - nOverlapX))
        if nBlockX == nTrace:
            nBy = 1

        orgBlock = np.zeros((nBlockT, nBlockX))

        GG = np.zeros((nGather * nBx * nBy, nBlockT, nBlockX, 1))

        for ii in range(nGather):

            for iBx in range(nBx):
                xStart = iBx * (nBlockT - nOverlapT) + 1
                xEnd = xStart + nBlockT - 1
                if xEnd > nPoint:
                    xEnd = nPoint
                    xStart = xEnd - nBlockT + 1
                for iBy in range(nBy):
                    yStart = (iBy) * (nBlockX - nOverlapX) + 1
                    yEnd = yStart + nBlockX - 1
                    if yEnd > nTrace:
                        yEnd = nTrace
                        yStart = yEnd - nBlockX + 1

                    orgBlock[:, :] = orgg[xStart - 1:xEnd, ii * nTrace + yStart - 1:ii * nTrace + yEnd]

                    GG[ii * nBx * nBy + iBx * nBy + iBy, :, :, 0] = orgBlock

        return GG

if __name__ == '__main__':
    mynet = myMDnet()  # # Instantiate class
    mynet.train()  # Method in startup class