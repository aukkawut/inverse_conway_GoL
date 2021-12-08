import numpy as np
from game import *
from pipeline import *
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf
import tensorflow.keras as keras


def evaluation_one(grid,outcome,dt):
    '''
    This function will evaluate score for one output game at timestep t + dt. We want to minimize the score.
    '''
    score = 0
    for i in range(dt):
        outcome = one_iter(outcome)
    for i in range(len(outcome)):
        for j in range(len(outcome[0])):
            if outcome[i][j] != grid[i][j]:
                score = score + 1
    return score

def evaluation_sanity_check():
    '''
    This function will evaluate the sanity check of the evaluation function
    '''
    grid = create_grid(10)
    pos = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    grid = fill_grid(pos, grid)

    grid2 = one_iter(one_iter(grid))

    assert evaluation_one(grid2, grid,2) == True #should be True
    assert evaluation_one(grid2, grid,1) == False #should be False

def test_train(data):
    '''
    This function will split the data into training and testing data
    '''
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,1:627], data.iloc[:,626:1251], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def CNN_pipeline(data):
    '''
    This function will split data into training and testing while making it into 2 channel matrix form
    '''
    X_train, X_test, y_train, y_test = test_train(data)
    X_train = X_train[:,1:626].values.reshape(X_train.shape[0],25,25)
    X_train2 = np.full((25,25),X_train[:,:,0])
    X_test = X_test.values.reshape(X_test.shape[0],25,25)
    return X_train, X_test, y_train, y_test

def FullyConnected():
    '''
    This function will create the FullyConnected NN model
    '''
    model = Sequential()
    model.add(layers.Input((626,)))
    model.add(layers.Dense(626, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(625, activation='relu'))
    model.add(layers.Dense(625, activation='softmax'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def train_FullyConnected(x_train, y_train, batch_size = 32, epochs = 10):
    '''
    This function will train the fully connected neural network model
    '''
    model = FullyConnected()
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return model, hist

def saveModel(model,name='./model/model.h5'):
    '''
    This function will save the model
    '''
    model.save(name)

def plot4matrices(grid,grid2,grid3,grid4):
    '''
    This function will plot the 3 matrices
    '''
    fig, axs = plt.subplots(1, 4, figsize=(15,5))
    axs[0].imshow(grid, cmap='gray_r', interpolation='nearest')
    axs[1].imshow(grid2, cmap='gray_r', interpolation='nearest')
    axs[2].imshow(grid3, cmap='gray_r', interpolation='nearest')
    axs[3].imshow(grid4, cmap='gray_r', interpolation='nearest')
    axs[0].title.set_text('Input')
    axs[1].title.set_text('Prediction')
    axs[2].title.set_text('Ground Truth')
    axs[3].title.set_text('Reconstruction')
    plt.show()

def fire_not_fire(grid,threshold = 0.01):
    '''
    This function will return the matrix in which for each value that greater than threshold will yield 1, otherwise 0
    '''
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i,j] > threshold:
                grid[i,j] = 1
            else:
                grid[i,j] = 0
    return grid   

def CNN():
    '''
    This function will create the CNN model
    %TODO:
        - Change the input shape to (1,625) with another input dt
        - Test and adjust the model
    '''
    in1 = layers.Input(shape=(625,))
    in2 = layers.Input(shape=(1,))
    x = layers.Reshape((25,25,1))(in1)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.flatten(x)
    y = layers.Dense(625, activation='relu',)(in2)
    x = layers.concatenate([x,y])
    x = layers.Dense(625, activation='relu')(x)
    x = layers.Dense(625, activation='relu')(x)
    out = layers.Dense(625, activation='softmax')(x)
    model = Model(inputs=[in1,in2], outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model
def prepare_CNN(data):
    '''
    This function will prepare the data for the CNN model
    '''
    X_train, X_test, y_train, y_test = test_train(data)
    X_train = X_train.iloc[:,1:626]
    X_train2 = X_train.iloc[:,:1]
    X_test = X_test.iloc[:,1:626]
    X_test2 = X_test.iloc[:,:1]
    return X_train, X_test, y_train, y_test, X_train2, X_test2
def train_CNN(x_train,x_train2, y_train, batch_size = 32, epochs = 10):
    '''
    This function will train the CNN model
    '''
    model = CNN()
    hist = model.fit([x_train,x_train2], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return model, hist
def get_generator():
    model = Sequential()
    model.add(layers.Input((625,)))
    model.add(layers.Dense(625, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(1, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(625, activation='softmax'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model.summary())
    return model
def get_discriminator():
    model = Sequential()
    model.add(layers.Input((625,)))
    model.add(layers.Dense(625, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(1, activation='softmax'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model.summary())
    return model

class CycleGan(keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }
        
class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 3, figsize=(12, 12))
        for i, img in enumerate(tf_X_test.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy().reshape((25,25))
            prediction = fire_not_fire(prediction,0.01)
            img = (img[0]).numpy().astype(np.uint8).reshape((25,25))
            gt = one_iter(prediction)
            ax[i, 0].imshow(img, cmap='gray_r', interpolation='nearest')
            ax[i, 1].imshow(prediction, cmap='gray_r', interpolation='nearest')
            ax[i, 2].imshow(gt, cmap='gray_r', interpolation='nearest')
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 2].set_title("Reconstruction")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
            ax[i, 2].axis("off")

            prediction = keras.preprocessing.image.array_to_img(prediction.reshape((25,25,1)))
            prediction.save(
                "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
            )
        plt.show()
        plt.close()

adv_loss_fn = keras.losses.MeanSquaredError()
# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


def GAN(x,y, steps_per_epoch = 625, epochs=10):
    '''
    This function will generate the convolutional cyclic generative adversarial network (CCycleGAN) 
    %TODO:
        - Write the function
    '''
    gen_G  = get_generator()
    gen_F = get_generator()
    disc_X = get_discriminator()
    disc_Y = get_discriminator()
    # Create cycle gan model
    cycle_gan_model = CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )
    # Callbacks
    plotter = GANMonitor()

    cycle_gan_model.fit(
        zip(x, y),
        steps_per_epoch = steps_per_epoch,
        epochs=epochs,
        callbacks=[plotter],
    )
    return cycle_gan_model, gen_G, gen_F, disc_X, disc_Y

def generate_data(num = 100000,batch_size = 32):
    '''
    This function will generate the num data
    '''
    x = []
    y = []
    for i in range(num):
        grid = create_grid(25)
        grid = fill_grid(random_points(25,np.random.randint(1,high = 75)), grid)
        y.append(grid.reshape(625,1))
        x.append(one_iter(grid).reshape(625,1))
    return tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x)).batch(batch_size), tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y)).batch(batch_size)