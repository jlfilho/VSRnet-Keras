import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Input, Conv2D, LSTM, TimeDistributed
from keras.layers import LeakyReLU, PReLU, ReLU, Lambda, Add, Concatenate
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import regularizers
from keras_tqdm import TQDMCallback

from util import DataLoader, plot_temporal_test_images, VideoRestore
from losses import VGGLoss, VGGLossNoActivation, ssim, euclidean, mse
from losses import psnr3 as psnr
#from subpixel import SubpixelConv2D

class VSRnet():

    def __init__(self,
                 height_lr=24, width_lr=24, channels=1,
                 upscaling_factor=4,
                 gen_lr=1e-4, dis_lr=1e-4,
                 # VGG scaled with 1/12.75 as in paper
                 loss_weights=[1e-3, 0.006],
                 training_mode=True,
                 media_type='vt',
                 time_step=3
                 ):


        # Media type: vt - video temporal
        self.media_type = media_type
        self.time_step = time_step

        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr

        # High-resolution image dimensions
        if upscaling_factor not in [2, 4, 8]:
            raise ValueError(
                'Upscaling factor must be either 2, 4, or 8. You chose {}'.format(upscaling_factor))
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)

        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)

        # Learning rates
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr

        # Scaling of losses
        self.loss_weights = loss_weights

        # Gan setup settings
        #self.VGGLossNoAct = VGGLossNoActivation(self.shape_hr)
        #self.VGGLoss = VGGLoss(self.shape_hr)
        self.gan_loss = 'mse'#self.VGGLossNoAct.plus_content_loss
        self.dis_loss = 'binary_crossentropy'

        # Build & compile the generator network
        self.cnn = self.build_cnn()
        self.compile_cnn(self.cnn)
    

    def save_weights(self, filepath):
        """Save the generator and discriminator networks"""
        self.cnn.save_weights(
            "{}_generator_{}X.h5".format(filepath, self.upscaling_factor))
        

    def load_weights(self, generator_weights=None, **kwargs):
        print(">> Loading weights...")
        if generator_weights:
            self.cnn.load_weights(generator_weights, **kwargs)
        
    
    def compile_cnn(self, model):
        """Compile the generator with appropriate optimizer"""
        
        model.compile(
            loss=self.gan_loss,
            optimizer=Adam(lr=self.gen_lr,beta_1=0.9, beta_2=0.999)
            #metrics=[self.VGGLossNoAct.mse_content_loss,mse,psnr,ssim]
        )

    def build_cnn(self, residual_blocks=16):

        
        def SubpixelConv2D(scale=2, name="subpixel"):

            def subpixel_shape(input_shape):
                dims = [input_shape[0],
                        None if input_shape[1] is None else input_shape[1] * scale,
                        None if input_shape[2] is None else input_shape[2] * scale,
                        int(input_shape[3] / (scale ** 2))]
                output_shape = tuple(dims)
                return output_shape

            def subpixel(x):
                return tf.depth_to_space(x, scale)

            return Lambda(subpixel, output_shape=subpixel_shape, name=name)

        
        def up_sampling_block(model, kernal_size, filters, strides, number):
            model = Conv2D(filters=filters, kernel_size=kernal_size,
                           strides=strides, padding="same")(model)
            model = SubpixelConv2D(name='upSampleSubPixel_' +
                                   str(number), scale=2)(model)
            model = ReLU()(model)
            return model

        
        inputs = []
        convs1 = []
        convs2 = []
        
        for i in range(self.time_step):
            inputs.append(Input(shape = (64, 64,1), name='input_'+str(i)))
            convs1.append(Conv2D(filters= 64, kernel_size = 9, activation='relu',
                       padding = "same", name='conv1_'+str(i))(inputs[i]))

            convs2.append(Conv2D(filters= 32, kernel_size = 5, activation='relu',
                       padding = "same", name='conv2_'+str(i))(convs1[i]))
        if self.time_step > 1:
            concats = Concatenate()(convs2)
        else:
            concats = convs2[0]
    
        out = Conv2D(filters= 1, kernel_size = 5, activation='relu',strides=1, padding = "same", name='conv3')(concats)
        
        model = Model(inputs=inputs, outputs=out)
        model.summary()
        return model


    def train(self,
            epochs=5,
            batch_size=8,
            steps_per_epoch=5,
            steps_per_validation=5,
            crops_per_image=4,
            print_frequency=1,
            log_tensorboard_update_freq=10,
            workers=4,
            max_queue_size=5,
            model_name='TSRGAN',
            datapath_train='../../../videos_harmonic/MYANMAR_2160p/train/',
            datapath_validation='../../../videos_harmonic/MYANMAR_2160p/validation/',
            datapath_test='../../../videos_harmonic/MYANMAR_2160p/test/',
            log_weight_path='../model/', 
            log_tensorboard_path='../logs/',
            log_test_path='../test/'
        ):

        # Create data loaders
        train_loader = DataLoader(
            datapath_train, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image,
            self.media_type,
            self.time_step
        )

        validation_loader = None 
        if datapath_validation is not None:
            validation_loader = DataLoader(
                datapath_validation, batch_size,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                crops_per_image,
                self.media_type,
                self.time_step
        )

        test_loader = None
        if datapath_test is not None:
            test_loader = DataLoader(
                datapath_test, 1,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                1,
                self.media_type,
                self.time_step
        )

        # Callback: tensorboard
        callbacks = []
        if log_tensorboard_path:
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_tensorboard_path, model_name),
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=True,
                write_grads=True,
                update_freq=log_tensorboard_update_freq
            )
            callbacks.append(tensorboard)
        else:
            print(">> Not logging to tensorboard since no log_tensorboard_path is set")

        # Callback: Stop training when a monitored quantity has stopped improving
        earlystopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, verbose=1, 
            restore_best_weights=True )
        callbacks.append(earlystopping)

        # Callback: Reduce lr when a monitored quantity has stopped improving
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, min_lr=2*1e-6)
        callbacks.append(reduce_lr)

        # Callback: save weights after each epoch
        modelcheckpoint = ModelCheckpoint(
            os.path.join(log_weight_path, model_name + '_{}X.h5'.format(self.upscaling_factor)), 
            monitor='val_loss', 
            save_best_only=True, 
            save_weights_only=True)
        callbacks.append(modelcheckpoint)
  
        # Callback: test images plotting
        if datapath_test is not None:
            testplotting = LambdaCallback(
                on_epoch_end=lambda epoch, logs: None if ((epoch+1) % print_frequency != 0 ) else plot_temporal_test_images(
                    self,
                    test_loader,
                    datapath_test,
                    log_test_path,
                    epoch+1,
                    name=model_name,
                    time_step=self.time_step))
        callbacks.append(testplotting)

        #callbacks.append(TQDMCallback())

        self.cnn.fit_generator(
            train_loader,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_loader,
            validation_steps=steps_per_validation,
            callbacks=callbacks,
            use_multiprocessing=workers>1,
            workers=workers
        )


# Run the TSRGAN network
if __name__ == "__main__":

    # Instantiate the TSRGAN object
    print(">> Creating the VSRnet network")
    vsrnet = VSRnet(height_lr=64, width_lr=64,gen_lr=1e-4,upscaling_factor=2,time_step=3)
    #vsrnet.load_weights(generator_weights="../model/TS3RDB-t3_allLeakRelu_restemp_n64k3s1_rrdbn64k3s1b2_rrdbn64k3s1b6_plusloss_v4_2X.h5")
    
    #restore = VideoRestore()
    #print(">> Generating video...")
    #restore.write_temporal_srvideo(model=ts3rrdb,lr_videopath='../../../videos_harmonic/MYANMAR_2160p/LR/myanmar01_320x180.mp4',sr_videopath="../test/myanmar01_SRx2.mp4",print_frequency=30,crf=15,time_step=3)
    #restore.write_temporal_srvideo(model=ts3rrdb,lr_videopath='../../../videos_harmonic/MYANMAR_2160p/4x/myanmar01_4x.mp4',sr_videopath="../test/myanmar01_2x.mp4",print_frequency=30,crf=15,time_step=3)

    vsrnet.train(
            epochs=10000,
            batch_size=8,
            steps_per_epoch=50,
            steps_per_validation=10,
            crops_per_image=8,
            print_frequency=1,
            log_tensorboard_update_freq=10,
            workers=1,
            max_queue_size=11,
            model_name='VSRNet',
            datapath_train='../../../videos_harmonic/MYANMAR_2160p/train/',
            datapath_validation='../../../videos_harmonic/MYANMAR_2160p/validation/',
            datapath_test='../../../videos_harmonic/MYANMAR_2160p/test/',
            log_weight_path='../model/', 
            log_tensorboard_path='../logs/',
            log_test_path='../test/'
    )