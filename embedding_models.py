def network1(image_size,kernel_size):
    #a sincle convolution
    orig = keras.Input(shape=(image_size, image_size, 3))
    x = layers.Conv2D(3, kernel_size=kernel_size, padding="same")(orig)
    inv_model = keras.Model(orig,x)
    return(inv_model)

def network2(image_size,ch,kernel_size,activ='swish'):
    orig = keras.Input(shape=(image_size, image_size, 3))
    x = layers.Conv2D(ch, kernel_size=kernel_size, padding="same",activation=activ)(orig)
    x = layers.Conv2D(ch, kernel_size=kernel_size, padding="same",activation=activ)(x)
    x = layers.Conv2D(3, kernel_size=kernel_size, padding="same")(x)
    inv_model = keras.Model(orig,x)
    return(inv_model)

def network3(image_size,ch,kernel_size,activ='swish'):
    orig = keras.Input(shape=(image_size, image_size, 3))
    x = layers.Conv2D(ch, kernel_size=kernel_size, padding="same",activation=activ)(orig)
    x = layers.Conv2D(ch, kernel_size=kernel_size, padding="same",activation=activ)(x)
    x = layers.Conv2D(ch, kernel_size=3, padding="same",activation=activ)(x)
    x = layers.Conv2D(ch, kernel_size=3, padding="same",activation=activ)(x)
    x = layers.Conv2D(3, kernel_size=3, padding="same")(x)
    inv_model = keras.Model(orig,x)
    return(inv_model)
