import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add, Concatenate, Lambda, UpSampling2D
from tensorflow.keras.models import Model


def RDB(x, n_layers = 3, growth_rate = 64, name = 'RDB'):
    
    # x: input feature map tensor 
    # n_layers: number of convolutional layers inside each RDB 
    # growth_rate: number of filter (feature map) per convolution
    # name: block name for layer naming (helps when stacking many blocks)
    
    inputs = x  # Save the origianl input for residual addition later 
    concat_layers = [x] # keep track of all feature maps for dense concatenation 
    
    # --- Dense Convolutional Connection --- 
    for i in range(n_layers):
        
        # 1. Apply a 3x3 convolution to the current feature map 
        x = Conv2D(growth_rate, 3, padding='same', name=f'{name}_con{i+1}')(x)
        
        
        # 2. Apply ReLU activation (introduces non-linearity)
        x = ReLU()(x)
        
        # 3, Append new featurue map to the list of dense connctions 
        concat_layers.append(x)
        
        # 4. Concatenate all previous outputs along the channel dimesion
        # allows later layers to access featurer from all prev layers
        x = Concatenate()(concat_layers)
    
    # --- Local Feature Fusion (LFF) ---
    # After dense connection, channel count has grown a lot 
    # use 1x1 convolution to fuse (compress) back to "groth-rate"
    
    x = Conv2D(growth_rate, 1, padding='same', name=f'{name}_LFF')(x)

    # --- local residual learning (LRL) ---
    # add original input back to the fused output 
    # helps gradient flow and prevents overfitting 
    # rediaul connection y = x + H(x) where x is input block 
    # and H(x) is learned transformation from the dense layers
    
    
    return Add()([x, inputs])


def build_basic_RDN(scale_w=2, scale_h=2, n_blocks=8, n_layers=3, growth_rate = 64, channels=15):
    # scale: upscaling factor 
    # n_blocks: number of RDB stackked sequentially 
    # n_layers: number of conv layers in each RDB 
    # growth_rate: number of convolutiopnal filters (feature maps) per layer 
    # channels: number of image channels
    
    # input tensor with (none, none) for spetial dimension (height x width)
    input_tensor = Input(shape=(None, None, channels))
    
    # --- Shallow Feature Extraction ---
    
    # Apllies 3x3 conv to extracrt shallow features to capture low-lvl deatils (edges, graditns, etc)
    x = Conv2D(growth_rate, 3, padding='same', name='shallow1')(input_tensor)
    # Refine features (give smoother feature maps)
    x = Conv2D(growth_rate, 3, padding='same', name='shallow2')(x)
    
    # save tensor as input for gloabl residual addition (where we skip connection)
    inputs = x
    
    # --- Residual Dense Block Stacck ---
    # each RDB_outs store output from all blocks for global feature fusion later 
    RDB_outs = []
    for i in range(n_blocks):
        # RDB() extract dense local features with residual learning 
        # output shape should be the same: (H, W, 64)
        x = RDB(x, n_layers=n_layers, growth_rate=growth_rate, name=f'RDB{i+1}')
        RDB_outs.append(x)
        
    
    # --- Global Feature Fusion (GFF) --- 
    
    # joints all RDB outputs along channel dimensions to combine information
    # from all residual blocks 
    x = Concatenate()(RDB_outs)
    
    # compress channels (local fusion) to reduce feature dimensionality 
    x = Conv2D(growth_rate, 1, padding='same', name='GFF_1')(x)
    
    # refine fused features to capture glocal context
    x = Conv2D(growth_rate, 3, padding='same', name='GFF_2')(x)
    
    # adds original shallow feature map 
    # implement global residual learning (stabilizes traning, preserves low_freq info)
    x = Add()([x, inputs])
        
    
    # --- Upscaling block (Bilinaer Interpolation for Brightness Temperature) --- 
    # The goal here is to enlarge spatial resolution of cont. BT fields
    # in a physical consistent (smooth) way. Here we use bilinear interpolation, 
    # since it produces transitions that is more representative of radiometric data 
    # due as it computes the weighted average based on distances and vary gradually
    # can be explained by atmospheric temp, water vapor, and emssivity cahnges.
    # There will be no abrupt jupms like pixel intensity in photographic images.
    
    # Bilinear upsampled
    # takes the input LR image tensor and produces a bilinearly upsampled base image
    # (same number of channels as input), scaled by (scale_h, scale_w)
    base = UpSampling2D(
                size=(scale_h, scale_w),
                interpolation="bilinear",
                name="bilinear_base"
            )(input_tensor)
    # At this point: base has shape (H * scale_h, W * scale_w, channels)
    # since base is computed from input_tensor directly, it preserves the radiometric values
    # preserves original LR signal characteristics while increasing spatial resolution smoothly.
    
    
    # Input Shape: (H, W, growth_rate)
    # Output Shape: (scale_h * H, scale_w * W, growth_rate)
    # simultaneous upscaling both dimensions
    x = UpSampling2D(size=(scale_h, scale_w), interpolation='bilinear')(x)
    
    # Refinemnet convolution 
    # smoothly refines the interpotaed BT field to correct small-scale biases
    # without over-sharpenning or creating artifical image
    x = Conv2D(channels, 3, padding='same', activation='linear', name='bt_refine')(x)
    
    # --- Residual Prediction ---
    # predict residual bt field to add back to the bilinear upsampled base
    pred = Conv2D(channels, 3, padding='same', activation='linear', name='output_residual')(x)
    
    # --- Final Reconstruction --- 
    # last 3x3 convolution maps feature maps to the final image output 
    # output shape = (H * scale_h, W * scale_w, channels)
    out = Add(name='sr_out')([pred, base])
    return Model(inputs=input_tensor, outputs=out)



#### Test Ideas #### 

def pixel_shuffle_block(x, scale=2, scale_w=None, scale_h=None, filters=64, name=None): 
    # x: input feature map tensor 
    # scale: upscaling factor (e.g., 2 for doubling resolution)
    # scale_w: horizontal upscaling factor (optional, overrides scale)
    # scale_h: vertical upscaling factor (optional, overrides scale)
    # filters: number of convolutional filters (feature maps) before pixel shuffle
    # name: optional name for the layer
    
    # Use scale_w and scale_h if provided, otherwise use scale
    if scale_w is None:
        scale_w = scale
    if scale_h is None:
        scale_h = scale
    
    # Pixel Shuffle requires equal scales
    if scale_w != scale_h:
        raise ValueError(f"PixelShuffle requires scale_w == scale_h, got scale_w={scale_w}, scale_h={scale_h}")
    
    block_size = scale_w
    
    # 1. Apply a convolution to increase the number of channels to filters * (block_size^2)
    x = Conv2D(filters * (block_size ** 2), 3, padding='same')(x)
    
    # 2. Pixel Shuffle (Depth to Space) to rearrange channels into spatial dimensions
    x = Lambda(lambda t: tf.nn.depth_to_space(t, block_size=block_size))(x)
    
    # 3. Apply ReLU activation
    x = ReLU()(x)
    
    return x


def build_RDN_pixelshuffle(scale_w=2, scale_h=2, n_blocks=8, n_layers=3, growth_rate = 64, channels=1):
    # scale: upscaling factor 
    # n_blocks: number of RDB stackked sequentially 
    # n_layers: number of conv layers in each RDB 
    # growth_rate: number of convolutiopnal filters (feature maps) per layer 
    # channels: number of image channels
    
    # input tensor with (none, none) for spetial dimension (height x width)
    input_tensor = Input(shape=(None, None, channels))
    
    # --- Shallow Feature Extraction ---
    
    # Apllies 3x3 conv to extracrt shallow features to capture low-lvl deatils (edges, graditns, etc)
    x = Conv2D(growth_rate, 3, padding='same', name='shallow1')(input_tensor)
    # Refine features (give smoother feature maps)
    x = Conv2D(growth_rate, 3, padding='same', name='shallow2')(x)
    
    # save tensor as input for gloabl residual addition (where we skip connection)
    inputs = x
    
    # --- Residual Dense Block Stacck ---
    # each RDB_outs store output from all blocks for global feature fusion later 
    RDB_outs = []
    for i in range(n_blocks):
        # RDB() extract dense local features with residual learning 
        # output shape should be the same: (H, W, 64)
        x = RDB(x, n_layers=n_layers, growth_rate=growth_rate, name=f'RDB{i+1}')
        RDB_outs.append(x)
        
    
    # --- Global Feature Fusion (GFF) --- 
    
    # joints all RDB outputs along channel dimensions to combine information
    # from all residual blocks 
    x = Concatenate()(RDB_outs)
    
    # compress channels (local fusion) to reduce feature dimensionality 
    x = Conv2D(growth_rate, 1, padding='same', name='GFF_1')(x)
    
    # refine fused features to capture glocal context
    x = Conv2D(growth_rate, 3, padding='same', name='GFF_2')(x)
    
    # adds original shallow feature map 
    # implement global residual learning (stabilizes traning, preserves low_freq info)
    x = Add()([x, inputs])
        
    
    # --- Upscaling block (Bilinaer Interpolation for Brightness Temperature) --- 
    
    base = UpSampling2D(
                size=(scale_h, scale_w),
                interpolation="bilinear",
                name="bilinear_base"
            )(input_tensor)
    # At this point: base has shape (H * scale_h, W * scale_w, channels)
    # since base is computed from input_tensor directly, it preserves the radiometric values
    # preserves original LR signal characteristics while increasing spatial resolution smoothly.
    
    # PixelShuffle: require equal vertical and horizontal scale
    if scale_h != scale_w:
        raise ValueError(f"PixelShuffle requires scale_h == scale_w, got scale_h={scale_h}, scale_w={scale_w}")
    scale = scale_h

    # Apply single pixel-shuffle upsampling (depth_to_space)
    x = pixel_shuffle_block(x, scale=scale, filters=growth_rate)
    
    # Refinemnet convolution 
    # smoothly refines the interpotaed BT field to correct small-scale biases
    # without over-sharpenning or creating artifical image
    # Use relu to fix checkerboard artifacts from pixel shuffle, but can also try linear activation to preserve radiometric values
    x = Conv2D(channels, 3, padding='same', activation='linear', name='post_upsample_refine')(x)
    
    # --- Residual Prediction ---
    # predict residual bt field to add back to the bilinear upsampled base
    pred = Conv2D(channels, 3, padding='same', activation='linear', name='output_residual')(x)
    
    # --- Final Reconstruction --- 
    # last 3x3 convolution maps feature maps to the final image output 
    # output shape = (H * scale_h, W * scale_w, channels)
    out = Add(name='sr_out')([pred, base])
    return Model(inputs=input_tensor, outputs=out)