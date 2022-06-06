def isValidDir(dirpath) :
    existingFiles = os.listdir(dirpath)
    requiredFiles = ['reference.mhd', 'reference.raw', 'oct.mhd', 'oct.raw']
    for iFile in requiredFiles :
        if iFile not in existingFiles :
            return False
    return True

def getFileNames(dirNames):
    xfiles = []
    yfiles = []
    for dirName in dirNames :
        for subDirName in sorted(os.listdir(os.path.join(dataDir,dirName))) :
            if isValidDir(os.path.join(dataDir,dirName,subDirName)) :
                xfiles.append(os.path.join(dataDir,dirName,subDirName,'oct.mhd'))
                yfiles.append(os.path.join(dataDir,dirName,subDirName,'reference.mhd'))
    return xfiles, yfiles



def read_mhd_image(filepath) :

    images = sitk.GetArrayFromImage(
                sitk.ReadImage(filepath, imageIO="MetaImageIO")
            )
    
    return images


def load_data(xfiles, yfiles):
    ds = tf.data.Dataset.from_tensor_slices((
                read_mhd_image(xfiles[0]),
                read_mhd_image(yfiles[0])
            ))
    for xfile, yfile in zip(xfiles[1:], yfiles[1:]):
        ds = ds.concatenate( 
                        tf.data.Dataset.from_tensor_slices((
                            read_mhd_image(xfile),
                            read_mhd_image(yfile)
                        ))
            )
    return ds


def display(display_list):
  plt.figure(figsize=(15, 15))
  
  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    
    if i == 0:
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    else:
        plt.imshow(tf.squeeze(display_list[i], axis=-1), vmin=0, vmax=3)
    plt.axis('off')
  plt.show()

def create_mask(mask):
    mask = tf.argmax(mask, axis=-1)
    mask = mask[..., tf.newaxis]
    return mask

def show_predictions(dataset=None, num=1):

    for image, mask in dataset.take(num):

        # predict mask from model
        pred_mask = model.predict(image, batch_size=1)
        
        # convert true mask and pred mask from one_hot to integers
        #mask = create_mask(mask)
        pred_mask = create_mask(pred_mask)
        # display for each elements of the batch
        for idx in range(image.shape[0]):
            display([image[idx], mask[idx], pred_mask[idx]])


rescaling_layer = tf.keras.Sequential([                               
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])


def expand_dims(image, label):
    return tf.expand_dims(image, axis=-1), tf.expand_dims(label, axis=-1) # tf.uinit8, tf.uinit8


def resize(image, mask, new_height=HEIGHT, new_width=WIDTH):

    # Resample image to new size with bilinear interpolation
    image = tf.image.resize(image, [new_height, new_width], method='bicubic')
    
    # Resample mask to new size with nearest neibour interpolation 
    mask = tf.image.resize(mask, [new_height, new_width], method='nearest')
    
    #pad_height = int((PADDED_HEIGHT-HEIGHT)/2)
    #pad_width = int((PADDED_WIDTH-WIDTH)/2)

    #if len(image.shape)==3:    
    #    paddings = tf.constant([[pad_height, pad_height],[pad_width, pad_width], [0, 0,]])
    #else:
    #    paddings = tf.constant([[0, 0,],[pad_height, pad_height],[pad_width, pad_width], [0, 0,]])
    #image = tf.pad(image, paddings, 'SYMMETRIC')
    
    return image, mask # tf.uinit8 , resizing doesn't change datatype


def rescale(image, label):
    return rescaling_layer(tf.cast(image, tf.float32)), label # tf.float32, tf.uinit8
    #return tf.image.per_image_standardization(image), label # tf.float32, tf.uinit8

def random_flip_left_right(image, mask):
    """
    Randomly flip an image and label horizontally (left to right).
    """

    flip_cond = tf.less(tf.random.uniform([], 0, 1.0), .5)

    image = tf.cond(flip_cond, lambda: tf.image.flip_left_right(image), lambda: image)
    mask = tf.cond(flip_cond, lambda: tf.image.flip_left_right(mask), lambda: mask)

    return image, mask

def random_flip_up_down(image, mask):
    """
    Randomly flip an image and label horizontally (left to right).
    """

    flip_cond = tf.less(tf.random.uniform([], 0, 1.0), .2)
    
    image = tf.cond(flip_cond, lambda: tf.image.flip_up_down(image), lambda: image)
    mask = tf.cond(flip_cond, lambda: tf.image.flip_up_down(mask), lambda: mask)

    return image, mask

def random_zoom(image, mask):
    
    # random zoom if b-scan has fluid
    if tf.equal(tf.reduce_sum(mask), 0):
        return image, mask

    # randomly decide to zoom or not
    do_random_zoom = tf.less(tf.random.uniform([], 0, 1.0), .5)

    if do_random_zoom:

        # randomly set zoom range
        zoom_height = tf.random.uniform([], 256, 512, dtype=tf.int32)
        zoom_width = tf.random.uniform([], 256, 512, dtype=tf.int32)

        # randomly select offsets for zooming
        offset_height = tf.random.uniform([], 0, HEIGHT-zoom_height, dtype=tf.int32)
        offset_width = tf.random.uniform([], 0, WIDTH-zoom_width, dtype=tf.int32)

        image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, zoom_height, zoom_width)
        mask = tf.image.crop_to_bounding_box(mask, offset_height, offset_width, zoom_height, zoom_width)
        
        image, mask = resize(image, mask)

    return image, mask

def random_rotate(image, label):
    
    # random choose angle from -10.0 to 10deg
    angle = tf.random.uniform([], -10.0, 10.0)

    # direction of rotation
    if tf.less(angle, 0):
        angle += 360.0

    # convert angle from degrees to radians
    angle = np.pi * angle / 180.0

    image = tfa.image.rotate(image, angle)
    label = tfa.image.rotate(label, angle)

    return image, label

def random_cutout(image, mask):
    
    # randomly choose mask size
    mask_height = tf.random.uniform([], minval=16, maxval=64, dtype = tf.int32)
    if mask_height % 2 != 0:
        mask_height -= 1
    mask_width = tf.random.uniform([], minval=16, maxval=64, dtype = tf.int32)
    if mask_width % 2 != 0:
        mask_width -= 1
    image_shape = image.shape
    if len(image_shape)==3:
        image = tf.expand_dims(image, axis=0)

    image = tfa.image.random_cutout(image, mask_size=[mask_height, mask_width], constant_values=0)

    if len(image_shape)==3:
        image = tf.squeeze(image, axis=0)

    return image, mask


def elastic_transform(image, mask, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    # Make image and mask 4D Tensor as required input argumnet tfa.image.interpolate_bilinear
    image = tf.expand_dims(image, axis=0)
    
    alpha=34
    sigma=4

    mask = tf.expand_dims(mask, axis=0)

    #image = tf.transpose(image, perm=[3, 0, 1, 2])
    # Assign random state, to apply same deformation for image and mask
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = mask.shape
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    
    x, y = tf.meshgrid(tf.range(shape[2]), tf.range(shape[1]))
    
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x = tf.expand_dims(x, axis=0)
    y = tf.expand_dims(y, axis=0)
    x = tf.expand_dims(x, axis=3)
    y = tf.expand_dims(y, axis=3)

    x_cord = tf.reshape(x+dx, (1, -1, 1))
    y_cord = tf.reshape(y+dy, (1, -1, 1))

    query_points = tf.concat([x_cord, y_cord], axis=-1)

    image_t = tf.reshape(tfa.image.interpolate_bilinear(image, query_points, indexing='xy'), image.shape)
    mask_t = tf.reshape(tfa.image.interpolate_bilinear(mask, query_points, indexing='xy'), shape)

    image_t = tf.squeeze(image_t, axis=0)
    #image_t = tf.transpose(image_t, perm=[1, 2, 3, 0])
    mask_t = tf.squeeze(mask_t, axis=0)

    return image_t, mask_t

def random_elastic_deform(image, mask):
    
    # Apply transformation on image
    if tf.less(tf.random.uniform([], 0, 1.0), .7) :
        mask = tf.cast(mask, tf.float32)
        image, mask = elastic_transform(image, mask, WIDTH * 2, WIDTH * 0.08)
        mask = tf.cast(mask, tf.uint8)
    
    return image, mask

def random_skip_non_fluid(image, mask):
    
    notFluid = tf.reduce_all(tf.equal(mask, tf.zeros_like(mask)))
    
    if tf.logical_and(notFluid ,tf.cast(tf.random.uniform([],0,4, tf.int32), tf.bool)):
        skip = False
    else :
        skip = True
        
    return skip

def skip_non_fluid(image, mask):
        
    return tf.logical_not(tf.reduce_all(tf.equal(mask, tf.zeros_like(mask))))

def skip_fluid(image, mask):
        
    return tf.reduce_all(tf.equal(mask, tf.zeros_like(mask)))

def random_brightness(image, mask, mindelta=0.1, maxdelta=0.1):
    
    random_choice = tf.cast(tf.random.uniform([], 0, 2, tf.int32), tf.bool)

    delta = tf.cond(
        random_choice, 
        lambda : tf.random.uniform([],0, mindelta),
        lambda : tf.random.uniform([],0, maxdelta)
    )
    
    image = tf.cond(
        random_choice,
        lambda: tf.where((image - delta) < delta, 0., image-delta),
        lambda: tf.where((1. - image) < delta, 1., image+delta)
    )

    return image, mask

def random_contrast(image, mask, mingamma=.5, maxgamma=1.5):
    
    gamma = tf.random.uniform([], mingamma, maxgamma)
    image = tf.image.adjust_gamma(image, gamma)

    return image, mask

def median_filter_2d(image, mask):

    return tfa.image.median_filter2d(image), mask

def gaussian_filter_2d(image, mask):

    return tfa.image.gaussian_filter2d(image), mask

def random_blur(image, mask):
    
    # Apply transformation on image
    if tf.less(tf.random.uniform([], 0, 1.0), .2) :
        filtersize = np.random.randint(1, 20) #tf.random.uniform([], 1, 20, tf.int32)
        sigma = np.random.uniform(1, 20) #tf.random.uniform([], 1, 20, tf.float32)
        image = tfa.image.gaussian_filter2d(image, filtersize, sigma)
    
    return image, mask

def random_noise(image, mask):

    if tf.less(tf.random.uniform([], 0, 1.0), .3) :
        image = image + tf.random.normal(image.shape, mean=0, stddev=0.2, dtype=image.dtype)
    
    return image, mask

def preprocessing(image, mask):

    #image = tfa.image.median_filter2d(image, 3)
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image)-tf.reduce_min(image))

    return image, mask

def overlay_patch(img, patch, i=0, j=30, alpha=1):
    img_shape = tf.shape(img)
    img_rows, img_cols = img_shape[0], img_shape[1]
    patch_shape = tf.shape(patch)
    patch_rows, patch_cols = patch_shape[0], patch_shape[1]
    i_end = i + patch_rows
    j_end = j + patch_cols
    # Mix patch: alpha from patch, minus alpha from image
    overlay = alpha * (patch - img[i:i_end, j:j_end])
    # Pad patch
    overlay_pad = tf.pad(overlay, [[i, img_rows - i_end], [j, img_cols - j_end], [0, 0]])
    # Make final image
    img_overlay = img + overlay_pad
    return img_overlay

def extrapolate(image, mask):
    patch = image
    image = tf.image.resize(image, (268, 356))
    image = tf.pad(overlay_patch(image, patch, 0, 30), paddings=tf.constant([[44, 44,], [0, 0], [0,0]]))
    return image, mask

def random_shrink(image, mask):

    if tf.less(tf.random.uniform([], 0, 1.0), .5):
        image = tf.image.resize(image, [128, 256], method='bicubic')
        image = tf.pad(image, paddings=tf.constant([[64,64],[0,0],[0,0]]))

    return image, mask

def get_label(image, mask):
    mask = 0 if tf.reduce_sum(mask)==0 else 1
    return image, mask
