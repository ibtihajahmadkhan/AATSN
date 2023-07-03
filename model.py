import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.initializers import *
from tensorflow.keras.activations import *
from loss import *
from metrics import *
from tensorflow.keras import mixed_precision


def Norm_B_L(input_layer, type_norm = 'BN'):
    if type_norm == 'BN':
        x = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(input_layer)
    if type_norm == 'LN':
        x = LayerNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), trainable=True)(input_layer)
    return x


def spa_atten(enc_feat, dec_feat):
    enc_line = AveragePooling3D(pool_size=2, strides=1, padding="same")(enc_feat)
    enc_line = MaxPooling3D(pool_size=2, strides=1, padding="same")(enc_line)
    enc_line = Conv3D(filters=1, kernel_size=1, activation=None, padding='same')(enc_line)

    dec_line = AveragePooling3D(pool_size=2, strides=1, padding="same")(dec_feat)
    dec_line = MaxPooling3D(pool_size=2, strides=1, padding="same")(dec_line)
    dec_line = Conv3D(filters=1, kernel_size=1, activation=None, padding='same')(dec_line)

    out = add([enc_line, dec_line])
    out = sigmoid(out)
    return out


def fus_atten_gate(enc_feat, dec_feat, filters):
    spa_aten_gate = spa_atten(enc_feat, dec_feat)
    mul = multiply([spa_aten_gate, enc_feat])
    cat = concatenate([mul, dec_feat], axis=-1)
    out = Conv3D(filters=filters, kernel_size=3, activation='relu', padding='same')(cat)
    out = Norm_B_L(input_layer=out, type_norm = 'BN')
    return out


def conv_block(_feat, filters):
    x = Conv3D(filters, kernel_size=3, activation='relu', padding='same', kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(_feat)
    x = Norm_B_L(input_layer=x, type_norm = 'BN')
    return x


def bottleneck_block(in_feat, filters):
    x = Conv3D(filters, kernel_size=3, activation='relu', padding='same', kernel_initializer='glorot_normal',
               bias_initializer=Constant(0.1))(in_feat)
    x = Norm_B_L(input_layer=x, type_norm = 'BN')
    return x


def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling3D()(in_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    x = multiply([in_block, x])
    return x
    
## AATSN ++
def AATSN_Plus(input_shape=(144, 144, 144, 2)):
    inCT = Input(input_shape, name='CT')
    inPT = Input(input_shape, name='PT')

    #filters = [32, 64, 128, 256, 512]
    filters = [16, 32, 64, 128, 256, 512]
    # filters = [24, 48, 96, 192, 384]
    ratio = int(filters[0]/2)

    in_CT = Conv3D(filters=filters[0], kernel_size=7, padding='same', activation='relu')(inCT)
    in_CT = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(in_CT)
    in_PT = Conv3D(filters=filters[0], kernel_size=7, padding='same', activation='relu')(inPT)
    in_PT = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(in_PT)

    x_ct = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(in_CT)
    x_ct = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_ct)
    x_ct1 = se_block(x_ct, ch=filters[1], ratio=ratio)
    pool_ct = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_ct1)

    x_ct = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_ct)
    x_ct = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_ct)
    x_ct2 = se_block(x_ct, ch=filters[2], ratio=ratio)
    pool_ct = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_ct2)

    x_ct = Conv3D(filters=filters[3], kernel_size=3, activation='relu', padding='same')(pool_ct)
    x_ct = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_ct)
    x_ct3 = se_block(x_ct, ch=filters[3], ratio=ratio)
    
    ##__________________________________________________________________________________________________________________

    x_pt = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(in_PT)
    x_pt = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_pt)
    x_pt = se_block(x_pt, ch=filters[1], ratio=ratio)
    pool_pt = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_pt)

    x_pt = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_pt)
    x_pt = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_pt)
    x_pt = se_block(x_pt, ch=filters[2], ratio=ratio)
    pool_pt = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_pt)

    x_pt = Conv3D(filters=filters[3], kernel_size=3, activation='relu', padding='same')(pool_pt)
    x_pt = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0),
                              momentum=0.5)(x_pt)
    x_pt = se_block(x_pt, ch=filters[3], ratio=ratio)
    pool_pt = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_pt)

    conv_pt = bottleneck_block(pool_pt, filters[3])
    
    ##__________________________________________________________________________________________________________________

    up_sam_2 = Conv3DTranspose(filters[3], 3, strides=2, padding='same')(conv_pt)
    up_conv2 = fus_atten_gate(x_ct3, up_sam_2, filters[2])

    up_sam_3 = Conv3DTranspose(filters[2], 3, strides=2, padding='same')(up_conv2)
    up_conv3 = fus_atten_gate(x_ct2, up_sam_3, filters[1])

    up_sam_4 = Conv3DTranspose(filters[1], 3, strides=2, padding='same')(up_conv3)
    up_conv4 = fus_atten_gate(x_ct1, up_sam_4, filters[0])

    cat = Conv3D(filters[0], kernel_size=3, strides=1, activation='relu', padding='same')(up_conv4)
    outputs = Conv3D(1, kernel_size=5, strides=1, activation='sigmoid', padding='same')(cat)

    model = Model(inputs=[inCT, inPT], outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=dicefocal, metrics=dice_score_metric, run_eagerly=False)
    model.summary()
    return model

## AATSN
def AATSN(input_shape=(144, 144, 144, 2)):
    inCT = Input(input_shape, name='CT')
    inPT = Input(input_shape, name='PT')

    #filters = [32, 64, 128, 256, 512]
    #filters = [28, 56, 112, 224, 384]
    #filters = [24, 48, 96, 192, 384]
    filters = [16, 32, 64, 128, 256]

    in_CT = Conv3D(filters=filters[0], kernel_size=7, padding='same', activation='relu')(inCT)
    in_CT = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0),
                               momentum=0.5)(in_CT)
    in_PT = Conv3D(filters=filters[0], kernel_size=7, padding='same', activation='relu')(inPT)
    in_PT = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0),
                               momentum=0.5)(in_PT)

    x_ct = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(in_CT)
    x_ct1 = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_ct)
    pool_ct = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_ct1)

    x_ct = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(pool_ct)
    x_ct2 = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_ct)
    pool_ct = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_ct2)

    x_ct = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_ct)
    x_ct3 = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_ct)
    
    ## _____________________________________________________________________________________________________________

    x_pt = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(in_PT)
    x_pt = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0),
                              momentum=0.5)(x_pt)
    pool_pt = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_pt)

    x_pt = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(pool_pt)
    x_pt = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(x_pt)
    x_pt = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0),
                              momentum=0.5)(x_pt)
    pool_pt = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_pt)

    x_pt = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_pt)
    x_pt = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(x_pt)
    x_pt = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0),
                              momentum=0.5)(x_pt)
    pool_pt = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_pt)

    conv_pt = bottleneck_block(pool_pt, filters[2])

    ## _____________________________________________________________________________________________________________
    
    up_sam_2 = Conv3DTranspose(filters[2], 3, strides=2, padding='same')(conv_pt)
    up_conv2 = fus_atten_gate(x_ct3, up_sam_2, filters[2])

    up_sam_3 = Conv3DTranspose(filters[1], 3, strides=2, padding='same')(up_conv2)
    up_conv3 = fus_atten_gate(x_ct2, up_sam_3, filters[1])

    up_sam_4 = Conv3DTranspose(filters[0], 3, strides=2, padding='same')(up_conv3)
    up_conv4 = fus_atten_gate(x_ct1, up_sam_4, filters[0])

    cat = Conv3D(filters[0], kernel_size=3, strides=1, activation='relu', padding='same')(up_conv4)
    outputs = Conv3D(1, kernel_size=9, strides=1, activation='sigmoid', padding='same')(cat)

    model = Model(inputs=[inCT, inPT], outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=dicefocal, metrics=dice_score_metric, run_eagerly=False)
    model.summary()

    return model


