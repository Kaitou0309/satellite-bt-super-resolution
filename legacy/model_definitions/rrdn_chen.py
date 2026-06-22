import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add, Concatenate, Lambda, UpSampling2D
from tensorflow.keras.models import Model
from .rdn_chen import RDB, pixel_shuffle_block

def RRDB_block(x, n_rdb=3, n_layers=3, growth_rate=32, name='RRDB'):
    # x: input feature map tensor
    # n_rdb: number of RDBs in the RRDB block
    # n_layers: number of convolutional layers inside each RDB
    # growth_rate: number of filters (feature maps) per convolution
    # name: block name for layer naming
    # Logic: Input -> RDB1 -> RDB2 -> RDB3 -> Scaled_Residual + Input

    inputs = x  # Save the original input for residual addition later


    # Stack multiple RDBs sequentially （typicall 3 for standard ESRGAN)
    for i in range(n_rdb):
        x = RDB(x, n_layers=n_layers, growth_rate=growth_rate, name=f'{name}_RDB{i+1}')

    # Scale the output of the RRDB block
    # Math logic: y = x + beta * F(x) where beta is typically 0.2 to stabilize training
    # F(x) is the transformation learned by the RRDB block
    # Here we use a Lambda layer to scale the tensor
    x = Lambda(lambda t: t * 0.2)(x)

    # Add the original input back to the output of the RRDB block
    # We return y = x + beta * F(x) 
    return Add()([x, inputs])

def build_RRDN(scale_w=2, scale_h=2, n_rrdb=1, n_rdb_per_block=3, n_conv_layers=3, growth_rate=32, channels=1):
    
    # scale: upscaling factor
    # n_rrdb: number of RRDB blocks stacked sequentially (number of rdb blocks in each RRDB)
    # n_rdb_layers: number of conv layers in each RDB
    # growth_rate: number of convolutional filters (feature maps) per layer
    # channels: number of image channels

    # input tensor with (none, none) for spatial dimension (height x width)
    input_tensor = Input(shape=(None, None, channels))

    # --- Shallow Feature Extraction ---
    # extract low-level features from raw image
    shallow_feature = Conv2D(growth_rate, kernel_size=3, padding='same', name='shallow_feat')(input_tensor)

    # --- Deep Feature Extraction via RRDB Blocks ---
    x = shallow_feature
    
    # Stack multiple RRDB blocks sequentially
    # In a full ESRGAN model, n_rrdb_blocks is typically 23.
    # Here we use fewer for simplicity and small dataset
    for i in range(n_rrdb):
        x = RRDB_block(x, 
                       n_rdb=n_rdb_per_block,
                       n_layers=n_conv_layers,
                       growth_rate=growth_rate, 
                       name=f'RRDB{i+1}'
        )

    # --- Trunk Convolution --- 
    # Global Feature Fusion
    # One convolution to fuse features from all RRDB blocks before adding the global residual
    x = Conv2D(growth_rate, kernel_size=3, padding='same', name='GFF_conv')(x)
    
    # -- Global Residual Learning ---
    # We add the original shallow features back to the output of the trunk conv
    # This forces the *entire* network to learn a residual transformation 
    # (Forces *entire* trunk of RRDBs to only learn the "texture difference")
    x = Add()([x, shallow_feature])  # Global residual learning

    # --- Upsampling --- 
    # For BT, we prefer Bilinear upsampling followed by conv to reduce artifacts
    # rather than pixel shuffle or transposed conv

    x = UpSampling2D(size=(scale_h, scale_w), interpolation='bilinear', name='Upsample')(x)
    
    # Refinement
    x = Conv2D(64, 3, padding='same', activation='linear', name='Upsample_conv')(x)

    # --- Reconstruction ---
    # Final output layer to reconstruct the high-resolution image
    # (Linear activation for z-scores since we're predicting continuous pixel values)
    output_tensor = Conv2D(channels, 3, padding='same', name='Reconstruction')(x)

    # Create the model
    model = Model(inputs=input_tensor, outputs=output_tensor, name='Minimal_RRDN')

    return model






#### Test Ideas #### 

def build_rrdn_pixelshuffle(scale_w=2, scale_h=2, n_rrdb=1, n_rdb_per_block=3, n_conv_layers=3, growth_rate=32, channels=1):
    # Build RRDN model with pixel shuffle upsampling
    # scale: upscaling factor
    # n_rrdb: number of RRDB blocks stacked sequentially (number of rdb blocks in each RRDB)
    # n_rdb_layers: number of conv layers in each RDB
    # growth_rate: number of convolutional filters (feature maps) per layer
    # channels: number of image channels

    # input tensor with (none, none) for spatial dimension (height x width)
    input_tensor = Input(shape=(None, None, channels))

    # --- Shallow Feature Extraction ---
    shallow_feature = Conv2D(growth_rate, kernel_size=3, padding='same', name='shallow_feat')(input_tensor)

    # --- Deep Feature Extraction via RRDB Blocks ---
    x = shallow_feature
    
    for i in range(n_rrdb):
        x = RRDB_block(x, 
                       n_rdb=n_rdb_per_block,
                       n_layers=n_conv_layers,
                       growth_rate=growth_rate, 
                       name=f'RRDB{i+1}'
        )

    # --- Trunk Convolution --- 
    x = Conv2D(growth_rate, kernel_size=3, padding='same', name='GFF_conv')(x)
    
    # -- Global Residual Learning ---
    x = Add()([x, shallow_feature])  # Global residual learning

    # --- Upsampling via Pixel Shuffle --- 
    x = pixel_shuffle_block(x, scale_w=scale_w, scale_h=scale_h, filters=growth_rate, name='PixelShuffle_Upsample')

    # Refinement convolution
    x = Conv2D(64, 3, padding='same', activation='linear', name='Upsample_conv')(x)

    # --- Reconstruction ---
    output_tensor = Conv2D(channels, 3, padding='same', name='Reconstruction')(x)

    return Model(inputs=input_tensor, outputs=output_tensor, name='RRDN_PixelShuffle')
