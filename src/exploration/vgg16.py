import tensorflow as tf


def get_vgg_encoder(input_height=224, input_width=224, pretrained='imagenet'):
    """
    It creates the VGG 16 model using Keras. By default the weights are loaded into the model for transfer learning.
    :param input_height:                        Input image height (int)
    :param input_width:                         Input image width (int)
    :param pretrained:                          if equals "imagenet" it will load the weights. Otherwise transfer
                                                learning wont be used.
    :return:
    """
    vgg16 = tf.keras.Sequential([])
    vgg16.add(tf.keras.layers.Input(shape=(input_height, input_width, 3)))

    # Block1
    vgg16.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
    vgg16.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    f1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
    vgg16.add(f1)

    # Block 2
    vgg16.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    vgg16.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    f2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
    vgg16.add(f2)

    # Block 3
    vgg16.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    vgg16.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    vgg16.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    f3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')
    vgg16.add(f3)

    # Block 4
    vgg16.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    vgg16.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    vgg16.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    f4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')
    vgg16.add(f4)

    # Block 5
    vgg16.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    vgg16.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    vgg16.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    f5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
    vgg16.add(f5)

    #if pretrained == 'imagenet':
    #    VGG_Weights_path = keras.utils.get_file(weights_url.split("/")[-1], weights_url)
    #    Model(img_input, x).load_weights(VGG_Weights_path)

    return vgg16, [f1, f2, f3, f4, f5]

vgg, layers = get_vgg_encoder()