from tensorflow.keras import Model

from tensorflow.keras.layers import ( 
    Concatenate, LeakyReLU, BatchNormalization, ELU, ReLU, PReLU, Conv1D, Conv2D, 
    Input, MaxPool2D, AveragePooling2D, UpSampling2D, Dropout, Add, Activation,
    Conv2DTranspose, Multiply, DepthwiseConv2D )

from tensorflow.keras.layers.experimental.preprocessing import Rescaling, Resizing, Normalization

from tensorflow.keras.regularizers import L1, L2, L1L2

from tensorflow.keras.initializers import HeNormal, HeUniform

from tensorflow_addons.layers  import GroupNormalization, WeightNormalization

def double_conv(x, filters):

    kernel_regularizer = None#L2(1e-5)
    n_groups = 32 if filters >= 32 else int(filters)
    
    for _ in range(2):
        x = (
                    Conv2D(filters, (3,3), 
                            padding='same', 
                            kernel_initializer='HeUniform',
                            use_bias=False,
                            kernel_regularizer = kernel_regularizer))(x)
        x = BatchNormalization(scale=False)(x, training=True)
        x = ReLU() (x)
    
    return x
    
def UNet():
    
    n_layers = 5
    filters = 64
    skips = []

    inputs = Input((None, None, 1))

    x = inputs
    x = Rescaling(1./255) (x)
    x = Normalization() (x)

    # downsampling
    
    for i in range(n_layers):
        # double convolution
        x = double_conv(x, filters)
        skips.append(x)
        # downsample
        x = MaxPool2D(pool_size=(2, 2))(x)
        filters *= 2
    
    # bottleneck
    x = double_conv(x, filters)

    # expansion path
    while skips:
        
        # up sampling
        x = UpSampling2D(interpolation='bilinear') (x)
        # skip connection
        x = Concatenate(axis=3) ([x, skips.pop()])
        filters /= 2
        # double convolution
        x = double_conv(x, filters)

    x = Conv2D(OUTPUT_CHANNELS, (1, 1), activation='softmax') (x)

    return Model(inputs=[inputs], outputs=x)
