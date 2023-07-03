import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.initializers import *
from tensorflow.keras.activations import *
from loss import *
from metrics import *
from tensorflow.keras import mixed_precision

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.
    The model shapes are multipled by the batch size, but the weights are not.
    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )
    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory


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
    #out = sigmoid(enc_line)

    dec_line = AveragePooling3D(pool_size=2, strides=1, padding="same")(dec_feat)
    dec_line = MaxPooling3D(pool_size=2, strides=1, padding="same")(dec_line)
    dec_line = Conv3D(filters=1, kernel_size=1, activation=None, padding='same')(dec_line)

    out = add([enc_line, dec_line])
    out = sigmoid(out)
    return out


# def spa_atten(enc_feat, dec_feat):
#     enc_line_1 = AveragePooling3D(pool_size=2, strides=1, padding="same")(enc_feat)
#     enc_line_1 = Conv3D(filters=1, kernel_size=1, activation=None, padding='same')(enc_line_1)
#
#     enc_line_2 = MaxPooling3D(pool_size=2, strides=1, padding="same")(enc_feat)
#     enc_line_2 = Conv3D(filters=1, kernel_size=1, activation=None, padding='same')(enc_line_2)
#
#     out = add([enc_line_1, enc_line_2])
#     out = sigmoid(out)
#     return out


def spa_cha_atten_gate(enc_feat, dec_feat, filters):

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
    #x = Conv3D(filters, kernel_size=3, activation='relu', padding='same', kernel_initializer='glorot_normal',
    #           bias_initializer=Constant(0.1))(x)
    x = Norm_B_L(input_layer=x, type_norm = 'BN')
    return x

def robust_residual_block(input, filt, ker_sz = 3):
    xa_0 = Conv3D(filters=filt, kernel_size=1, padding='same', activation='relu')(input)
    bn_0 = Norm_B_L(input_layer=xa_0, type_norm = 'BN')

    xa_1 = Conv3D(filters=filt, kernel_size=3, padding='same', activation='relu')(bn_0)
    #bn_1 = Norm_B_L(input_layer=xa_1, type_norm = 'BN')

    xa_2 = Conv3D(filters=filt, kernel_size=1, padding='same', activation='relu')(xa_1)
    bn_2 = Norm_B_L(input_layer=xa_2, type_norm = 'BN')

    xb_0 = Conv3D(filters=filt, kernel_size=ker_sz, padding='same', activation='relu')(input)
    bn_3 = Norm_B_L(input_layer=xb_0, type_norm = 'BN')

    x = concatenate([bn_2, bn_3])
    x = Conv3D(filters=filt, kernel_size=3, activation='relu', padding='same')(x)
    x = Norm_B_L(input_layer=x, type_norm = 'BN')
    return x


def residual_block_Simp(input, filt):
    # xa_0 = Conv3D(filters=filt, kernel_size=1, padding='same', activation='relu')(input)
    # xa_0 = tf.keras.layers.LayerNormalization()(xa_0)
    #
    # xa_1 = Conv3D(filters=filt, kernel_size=3, padding='same', activation='relu')(xa_0)
    # xa_2 = Conv3D(filters=filt, kernel_size=1, padding='same', activation='relu')(xa_1)
    # xa_2 = tf.keras.layers.LayerNormalization()(xa_2)
    #
    # xb_0 = Conv3D(filters=filt, kernel_size=1, padding='same', activation='relu')(input)
    # xb_0 = tf.keras.layers.LayerNormalization()(xb_0)
    #
    # x = add([xa_2, xb_0])
    # x = Conv3D(filters=filt, kernel_size=3, activation='relu', padding='same')(x)
    # x = tf.keras.layers.LayerNormalization()(x)

    dec_line = GlobalAveragePooling3D()(input)
    print(tf.shape(dec_line ))
    dec_line = Dense(filters=filt, kernel_size=1, padding='same', activation='relu')(dec_line)
    x = concatenate([dec_line, input])

    return x

def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling3D()(in_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    # x = multiply()([in_block, x])
    x = multiply([in_block, x])
    return x


def AATSN_Plus(input_shape=(144, 144, 144, 2)):
    # input = Input(input_shape, name='Input')
    inCT = Input(input_shape, name='CT')
    inPT = Input(input_shape, name='PT')

    #filters = [32, 64, 128, 256, 512]
    # filters = [16, 32, 64, 128, 256, 512]
    # filters = [24, 48, 96, 192, 384]
    filters = [30, 60, 120, 240, 384]
    # filters = [28, 56, 112, 224, 384]
    # filters = [20, 40, 80, 120, 256, 512]
    ratio = int(filters[0]/2)

    in_CT = Conv3D(filters=filters[0], kernel_size=7, padding='same', activation='relu')(inCT)
    in_CT = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(in_CT)
    in_PT = Conv3D(filters=filters[0], kernel_size=7, padding='same', activation='relu')(inPT)
    in_PT = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(in_PT)

    x_ct = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(in_CT)
    x_ct = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_ct)
    x_ct1 = se_block(x_ct, ch=filters[0], ratio=ratio)
    pool_ct = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_ct1)

    x_ct = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(pool_ct)
    x_ct = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_ct)
    x_ct2 = se_block(x_ct, ch=filters[1], ratio=ratio)
    pool_ct = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_ct2)

    x_ct = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_ct)
    x_ct = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_ct)
    x_ct3 = se_block(x_ct, ch=filters[2], ratio=ratio)
    ##__________________________________________________________________________________________________________________

    x_pt = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(in_PT)
    x_pt = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_pt)
    x_pt = se_block(x_pt, ch=filters[0], ratio=ratio)
    pool_pt = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_pt)

    x_pt = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(pool_pt)
    #x_pt = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(x_pt)
    x_pt = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_pt)
    x_pt = se_block(x_pt, ch=filters[1], ratio=ratio)
    pool_pt = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_pt)

    x_pt = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_pt)
    #x_pt = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(x_pt)
    x_pt = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0),
                              momentum=0.5)(x_pt)
    x_pt = se_block(x_pt, ch=filters[2], ratio=ratio)
    pool_pt = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_pt)

    conv_pt = bottleneck_block(pool_pt, filters[2])

    up_sam_2 = Conv3DTranspose(filters[2], 3, strides=2, padding='same')(conv_pt)
    up_conv2 = spa_cha_atten_gate(x_ct3, up_sam_2, filters[2])

    up_sam_3 = Conv3DTranspose(filters[1], 3, strides=2, padding='same')(up_conv2)
    up_conv3 = spa_cha_atten_gate(x_ct2, up_sam_3, filters[1])

    up_sam_4 = Conv3DTranspose(filters[0], 3, strides=2, padding='same')(up_conv3)
    up_conv4 = spa_cha_atten_gate(x_ct1, up_sam_4, filters[0])

    cat = Conv3D(filters[0], kernel_size=3, strides=1, activation='relu', padding='same')(up_conv4)

    outputs = Conv3D(1, kernel_size=5, strides=1, activation='sigmoid', padding='same')(cat)

    model = Model(inputs=[inCT, inPT], outputs=outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=dicefocal, metrics=dice_score_metric, run_eagerly=False)

    model.summary()

    return model

# AATSN_Plus(input_shape=(144, 144, 144, 2))

def AATSN_Plus_16(input_shape=(144, 144, 144, 2)):
    # input = Input(input_shape, name='Input')
    inCT = Input(input_shape, name='CT')
    inPT = Input(input_shape, name='PT')

    #filters = [32, 64, 128, 256, 512]
    filters = [16, 32, 64, 128, 256, 512]
    #filters = [20, 40, 80, 120, 256, 512]
    # filters = [24, 48, 96, 192, 384]
    #filters = [30, 60, 120, 240, 384]
    #filters = [28, 56, 112, 224, 384]
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
    #x_pt = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(x_pt)
    x_pt = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_pt)
    x_pt = se_block(x_pt, ch=filters[2], ratio=ratio)
    pool_pt = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_pt)

    x_pt = Conv3D(filters=filters[3], kernel_size=3, activation='relu', padding='same')(pool_pt)
    #x_pt = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(x_pt)
    x_pt = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0),
                              momentum=0.5)(x_pt)
    x_pt = se_block(x_pt, ch=filters[3], ratio=ratio)
    pool_pt = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_pt)

    conv_pt = bottleneck_block(pool_pt, filters[3])

    up_sam_2 = Conv3DTranspose(filters[3], 3, strides=2, padding='same')(conv_pt)
    up_conv2 = spa_cha_atten_gate(x_ct3, up_sam_2, filters[2])

    up_sam_3 = Conv3DTranspose(filters[2], 3, strides=2, padding='same')(up_conv2)
    up_conv3 = spa_cha_atten_gate(x_ct2, up_sam_3, filters[1])

    up_sam_4 = Conv3DTranspose(filters[1], 3, strides=2, padding='same')(up_conv3)
    up_conv4 = spa_cha_atten_gate(x_ct1, up_sam_4, filters[0])

    cat = Conv3D(filters[0], kernel_size=3, strides=1, activation='relu', padding='same')(up_conv4)

    outputs = Conv3D(1, kernel_size=5, strides=1, activation='sigmoid', padding='same')(cat)

    model = Model(inputs=[inCT, inPT], outputs=outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=dicefocal, metrics=dice_score_metric, run_eagerly=False)

    model.summary()

    return model


def AATSN(input_shape=(144, 144, 144, 2)):
    # input = Input(input_shape, name='Input')
    inCT = Input(input_shape, name='CT')
    inPT = Input(input_shape, name='PT')

    #filters = [32, 64, 128, 256, 512]
    #filters = [24, 48, 96, 192, 384]
    filters = [28, 56, 112, 224, 384]

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
    #x_ct = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(x_ct)
    x_ct2 = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_ct)
    pool_ct = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_ct2)

    x_ct = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_ct)
    #x_ct = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(x_ct)
    x_ct3 = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0), momentum=0.5)(x_ct)
    #pool_ct = MaxPooling3D((2, 2, 2), strides=2, padding="same")(x_ct3)
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


    #conv_ct = bottleneck_block(pool_ct, filters[2])
    conv_pt = bottleneck_block(pool_pt, filters[2])

    #cat_all = concatenate([conv_ct, conv_pt], axis=-1)

    up_sam_2 = Conv3DTranspose(filters[2], 3, strides=2, padding='same')(conv_pt)
    up_conv2 = spa_cha_atten_gate(x_ct3, up_sam_2, filters[2])

    up_sam_3 = Conv3DTranspose(filters[1], 3, strides=2, padding='same')(up_conv2)
    up_conv3 = spa_cha_atten_gate(x_ct2, up_sam_3, filters[1])

    up_sam_4 = Conv3DTranspose(filters[0], 3, strides=2, padding='same')(up_conv3)
    up_conv4 = spa_cha_atten_gate(x_ct1, up_sam_4, filters[0])

    cat = Conv3D(filters[0], kernel_size=3, strides=1, activation='relu', padding='same')(up_conv4)
    #cat = Conv3D(filters[0], kernel_size=3, strides=1, activation='relu', padding='same')(cat)

    outputs = Conv3D(1, kernel_size=9, strides=1, activation='sigmoid', padding='same')(cat)

    model = Model(inputs=[inCT, inPT], outputs=outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=dicefocal, metrics=dice_score_metric, run_eagerly=False)

    model.summary()

    return model




def UNet_SE(input_shape=(144, 144, 144, 2)):
    # input = Input(input_shape, name='Input')
    in_CT = Input(input_shape, name='CT')
    in_PT = Input(input_shape, name='PT')

    filters = [24, 48, 96, 192, 384]
    ratio = 12

    cat_in = concatenate([in_CT, in_PT], axis=-1)

    r_conv1 = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(cat_in)
    r_conv1 = se_block(r_conv1, ch=filters[0], ratio=ratio)
    pool_1 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv1)

    r_conv2 = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(pool_1)
    r_conv2 = se_block(r_conv2, ch=filters[1], ratio=ratio)
    pool_2 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv2)

    r_conv3 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_2)
    r_conv3 = se_block(r_conv3, ch=filters[2], ratio=ratio)
    pool_3 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv3)

    # r_conv4 = robust_residual_block(pool, filters[3])
    # pool = MaxPooling3D( (2, 2, 2), strides=2, padding="same")(r_conv4)

    r_conv4 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_3)
    pool_4 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv4)
    r_conv5 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_4)

    # up_sam_1 = Conv3DTranspose(filters[3], 3, strides=2, padding='same')(r_conv5)
    # up_conv1 = spa_cha_atten_gate(r_conv4, up_sam_1, filters[3])

    up_sam_1 = Conv3DTranspose(filters[2], 3, strides=2, padding='same')(r_conv5)
    merge_1 = concatenate([pool_3, up_sam_1], axis=-1)
    up_conv_1 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(merge_1)

    up_sam_2 = Conv3DTranspose(filters[2], 3, strides=2, padding='same')(up_conv_1)
    merge_2 = concatenate([pool_2, up_sam_2], axis=-1)
    up_conv_2 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(merge_2)

    up_sam_3 = Conv3DTranspose(filters[1], 3, strides=2, padding='same')(up_conv_2)
    merge_3 = concatenate([pool_1, up_sam_3], axis=-1)
    up_conv_3 = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(merge_3)

    up_sam_4 = Conv3DTranspose(filters[0], 3, strides=2, padding='same')(up_conv_3)
    merge_4 = concatenate([r_conv1, up_sam_4], axis=-1)
    up_conv_4 = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(merge_4)

    cat = Conv3D(filters[0], kernel_size=3, strides=1, activation='relu', padding='same')(up_conv_4)

    outputs = Conv3D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(cat)

    model = Model(inputs=[in_CT, in_PT], outputs=outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=dicefocal, metrics=dice_score_metric,
                  run_eagerly=False)

    model.summary()

    return model

#UNet_SE(input_shape=(144, 144, 144, 2))


def residual_block(in_feat, filt, ker_sz = 3):
    bn_0 = Norm_B_L(input_layer=in_feat, type_norm = 'BN')
    xa_0 = Conv3D(filters=filt, kernel_size=3, padding='same', activation='relu')(bn_0)
    bn_1 = Norm_B_L(input_layer=xa_0, type_norm='BN')
    xa_1 = Conv3D(filters=filt, kernel_size=3, padding='same', activation='relu')(bn_1)

    xb_0 = Conv3D(filters=filt, kernel_size=1, padding='same', activation='relu')(in_feat)

    x = add([xa_1, xb_0])
    return x

def residual_decoder(dec, enc, filt):
    u_sam = UpSampling3D()(dec)
    x = concatenate([u_sam, enc])

    addr = residual_block(x, filt=filt)
    return addr

def Res_UNet(input_shape=(144, 144, 144, 2)):
    in_CT = Input(input_shape, name='CT')
    in_PT = Input(input_shape, name='PT')

    filters = [24, 48, 96, 192, 384]

    cat_in = concatenate([in_CT, in_PT], axis=-1)

    r_conv1 = residual_block(cat_in, filt=filters[0])
    pool_1 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv1)

    r_conv2 = residual_block(pool_1, filt=filters[1])
    pool_2 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv2)

    r_conv3 = residual_block(pool_2, filt=filters[2])
    pool_3 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv3)

    r_conv4 = residual_block(pool_3, filt=filters[3])
    pool_4 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv4)

    res_dec_1 = residual_decoder(pool_4, r_conv4, filt=filters[3])

    res_dec_2 = residual_decoder(res_dec_1, r_conv3, filt=filters[2])

    res_dec_3 = residual_decoder(res_dec_2, r_conv2, filt=filters[1])

    res_dec_4 = residual_decoder(res_dec_3, r_conv1, filt=filters[0])

    cat = Conv3D(filters[0], kernel_size=3, strides=1, activation='relu', padding='same')(res_dec_4)

    outputs = Conv3D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(cat)

    model = Model(inputs=[in_CT, in_PT], outputs=outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.binary_crossentropy, metrics=dice_score_metric, run_eagerly=False)

    model.summary()

    return model

#Res_UNet(input_shape=(144, 144, 144, 2))



def UNet_3D(input_shape=(144, 144, 144, 2)):
    # input = Input(input_shape, name='Input')
    in_CT = Input(input_shape, name='CT')
    in_PT = Input(input_shape, name='PT')

    filters = [16, 32, 64, 1128, 256]
    ratio = 12

    cat_in = concatenate([in_CT, in_PT], axis=-1)

    r_conv1 = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(cat_in)
    r_conv1 = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(r_conv1)
    pool_1 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv1)

    r_conv2 = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(pool_1)
    r_conv2 = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(r_conv2)
    pool_2 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv2)

    r_conv3 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_2)
    r_conv3 = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(r_conv3)
    pool_3 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv3)

    # r_conv4 = robust_residual_block(pool, filters[3])
    # pool = MaxPooling3D( (2, 2, 2), strides=2, padding="same")(r_conv4)

    r_conv4 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_3)
    pool_4 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv4)
    r_conv5 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_4)

    # up_sam_1 = Conv3DTranspose(filters[3], 3, strides=2, padding='same')(r_conv5)
    # up_conv1 = spa_cha_atten_gate(r_conv4, up_sam_1, filters[3])

    up_sam_1 = Conv3DTranspose(filters[2], 3, strides=2, padding='same')(r_conv5)
    merge_1 = concatenate([pool_3, up_sam_1], axis=-1)
    up_conv_1 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(merge_1)

    up_sam_2 = Conv3DTranspose(filters[2], 3, strides=2, padding='same')(up_conv_1)
    merge_2 = concatenate([pool_2, up_sam_2], axis=-1)
    up_conv_2 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(merge_2)

    up_sam_3 = Conv3DTranspose(filters[1], 3, strides=2, padding='same')(up_conv_2)
    merge_3 = concatenate([pool_1, up_sam_3], axis=-1)
    up_conv_3 = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(merge_3)

    up_sam_4 = Conv3DTranspose(filters[0], 3, strides=2, padding='same')(up_conv_3)
    merge_4 = concatenate([r_conv1, up_sam_4], axis=-1)
    up_conv_4 = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(merge_4)

    cat = Conv3D(filters[0], kernel_size=3, strides=1, activation='relu', padding='same')(up_conv_4)

    outputs = Conv3D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(cat)

    model = Model(inputs=[in_CT, in_PT], outputs=outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=dicefocal, metrics=dice_score_metric,
                  run_eagerly=False)

    model.summary()

    return model

#UNet_3D(input_shape=(144, 144, 144, 2))



def UNet_SE_3D(input_shape=(144, 144, 144, 2)):
    # input = Input(input_shape, name='Input')
    in_CT = Input(input_shape, name='CT')
    in_PT = Input(input_shape, name='PT')

    filters = [16, 32, 64, 128, 256]
    ratio = 12

    cat_in = concatenate([in_CT, in_PT], axis=-1)

    r_conv1 = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(cat_in)
    r_conv1 = se_block(r_conv1, ch=filters[0], ratio=ratio)
    pool_1 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv1)

    r_conv2 = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(pool_1)
    r_conv2 = se_block(r_conv2, ch=filters[1], ratio=ratio)
    pool_2 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv2)

    r_conv3 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_2)
    r_conv3 = se_block(r_conv3, ch=filters[2], ratio=ratio)
    pool_3 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv3)

    # r_conv4 = robust_residual_block(pool, filters[3])
    # pool = MaxPooling3D( (2, 2, 2), strides=2, padding="same")(r_conv4)

    r_conv4 = Conv3D(filters=filters[3], kernel_size=3, activation='relu', padding='same')(pool_3)
    pool_4 = MaxPooling3D((2, 2, 2), strides=2, padding="same")(r_conv4)
    r_conv5 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(pool_4)

    # up_sam_1 = Conv3DTranspose(filters[3], 3, strides=2, padding='same')(r_conv5)
    # up_conv1 = spa_cha_atten_gate(r_conv4, up_sam_1, filters[3])

    up_sam_1 = Conv3DTranspose(filters[3], 3, strides=2, padding='same')(r_conv5)
    merge_1 = concatenate([pool_3, up_sam_1], axis=-1)
    up_conv_1 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(merge_1)

    up_sam_2 = Conv3DTranspose(filters[2], 3, strides=2, padding='same')(up_conv_1)
    merge_2 = concatenate([pool_2, up_sam_2], axis=-1)
    up_conv_2 = Conv3D(filters=filters[2], kernel_size=3, activation='relu', padding='same')(merge_2)

    up_sam_3 = Conv3DTranspose(filters[1], 3, strides=2, padding='same')(up_conv_2)
    merge_3 = concatenate([pool_1, up_sam_3], axis=-1)
    up_conv_3 = Conv3D(filters=filters[1], kernel_size=3, activation='relu', padding='same')(merge_3)

    up_sam_4 = Conv3DTranspose(filters[0], 3, strides=2, padding='same')(up_conv_3)
    merge_4 = concatenate([r_conv1, up_sam_4], axis=-1)
    up_conv_4 = Conv3D(filters=filters[0], kernel_size=3, activation='relu', padding='same')(merge_4)

    cat = Conv3D(filters[0], kernel_size=3, strides=1, activation='relu', padding='same')(up_conv_4)

    outputs = Conv3D(1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(cat)

    model = Model(inputs=[in_CT, in_PT], outputs=outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.binary_crossentropy, metrics=dice_score_metric,
                  run_eagerly=False)

    model.summary()

    return model
