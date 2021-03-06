import sys  
import os
import gc
import math
import numpy as np
import cv2
import glob
import imageio
from PIL import Image
from random import choice
from keras.utils import Sequence
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Lambda, Input
from keras.models import Model
import tensorflow as tf
from subprocess import Popen, PIPE
from timeit import default_timer as timer
from losses import psnr2 as psnr

class Metrics():
    def __init__(self,use_gpu=True):
        '''
        net - ['squeeze','alex','vgg']
        '''

    def psnr_video(self,videopath1, videopath2):
        cap1 = cv2.VideoCapture(videopath1)
        cap2 = cv2.VideoCapture(videopath2)
        psnr_values = []
        while(cap1.isOpened() and cap2.isOpened()):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if(ret1 and ret2):
                psnr_values.append(self.calculate_psnr(frame1, frame2))        
            else:
                break
        cap1.release()
        cap2.release()
        return np.mean(psnr_values)
    
    def ssim_video(self,videopath1, videopath2):
        cap1 = cv2.VideoCapture(videopath1)
        cap2 = cv2.VideoCapture(videopath2)
        ssim_values = []
        while(cap1.isOpened() and cap2.isOpened()):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if(ret1 and ret2):
                ssim_values.append(self.calculate_ssim(frame1, frame2))        
            else:
                break
        cap1.release()
        cap2.release()
        return np.mean(ssim_values)

    # Calculate psnr between two images
    def calculate_psnr(self,img1, img2):
        # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))


    # Aux to Calculate psnr between two images
    def ssim(self,img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    # Calculate ssim between two images
    def calculate_ssim(self,img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self.ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self.ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self.ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')


class DataLoader(Sequence):
    def __init__(self, datapath, batch_size, height_hr, width_hr, scale, crops_per_image, media_type,time_step):
        """        
        :param string datapath: filepath to training images
        :param int height_hr: Height of high-resolution images
        :param int width_hr: Width of high-resolution images
        :param int height_hr: Height of low-resolution images
        :param int width_hr: Width of low-resolution images
        :param int scale: Upscaling factor
        """

        # Store the datapath
        self.datapath = datapath
        self.batch_size = batch_size
        self.height_hr = height_hr
        self.height_lr = int(height_hr / scale)
        self.width_hr = width_hr
        self.width_lr = int(width_hr / scale)
        self.channel = 1
        self.scale = scale
        self.crops_per_image = crops_per_image
        self.media_type  = media_type
        self.time_step=time_step
        self.total_imgs = None
        
        # Options for resizing
        self.options = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
        
        # Check data source
        self.img_paths = []

        if os.path.isdir(self.datapath):
            self.get_paths()
    
    def get_paths(self):
        for dirpath, _, filenames in os.walk(self.datapath):
            for filename in [f for f in filenames if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg','mp4','264','webm','wma'])]:
                self.img_paths.append(os.path.join(dirpath, filename))
        self.total_imgs = len(self.img_paths)
        print(">> Found {} images in dataset".format(self.total_imgs))
    
    def random_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]

    def fix_crop(self, img, dy, dx, y, x):
        return img[y:(y+dy), x:(x+dx), :]

    @staticmethod
    def scale_lr_imgs(imgs):
        """Scale low-res images prior to passing to SRGAN"""
        return imgs / 255.
    
    @staticmethod
    def unscale_lr_imgs(imgs):
        """Un-Scale low-res images"""
        return imgs * 255
    
    @staticmethod
    def scale_hr_imgs(imgs):
        """Scale high-res images prior to passing to SRGAN"""
        return imgs / 127.5 - 1
    
    @staticmethod
    def unscale_hr_imgs(imgs):
        """Un-Scale high-res images"""
        return (imgs + 1.) * 127.5
    
    def count_frames_manual(self,cap):
        count=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if(ret):
                count +=1
            else:
                break
        return count
    
    def count_frames(self,cap):
        '''Count total frames in video'''
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            total = self.count_frames_manual(cap)
        return total


    def get_random_frames(self,n_fms=1,cap=None,time_step=1):
        """Get random number of video frames"""
        np.random.seed(int(1000000*(timer()%1)))
        t_frames = self.count_frames(cap)
        if(time_step==1):
            choiced_frames = np.random.randint(t_frames, size=n_fms)
        else:
            choiced_frames = np.random.randint(t_frames-time_step, size=n_fms)
        return choiced_frames
    
    @staticmethod
    def load_img(path):
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)


    def load_frame(self,videopath,time_step=1,colorspace='YCbCr'):
        """Get n_imgs random frames from the video"""
        cap = cv2.VideoCapture(videopath)
        if(not cap.isOpened()):
            print("Error to open video: ",videopath)
            return -1 
        choiced_frame=self.get_random_frames(1,cap,time_step)
        cap.set(1,choiced_frame)
        frames = []
        for i in list(range(time_step)):
            ret, frame = cap.read()
            if not ret:
                print(">> Erro to access frames.")
                return -1
            if colorspace=='YCbCr':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:,:,:self.channel]
                frames.append(frame)
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
        cap.release()
        return np.array(frames)    


    def resize(self,shape,scale,image):
    
        # 3 channel images of arbitrary shape
        inp = Input(shape=shape)
        try:
            out = Lambda(lambda image: K.tf.image.resize_bicubic(image, (shape[0]//scale, shape[1]//scale)))(inp)
        except :
            # if you have older version of tensorflow
            out = Lambda(lambda image: K.tf.image.resize_bicubic(image, shape[0]//scale, shape[1]//scale))(inp)
        model = Model(inputs=inp, outputs=out)
        out = model.predict(image[np.newaxis, ...])
        return np.squeeze(out, axis=0).astype(np.uint8) 
   
    
    def __len__(self):
        return int(self.total_imgs / float(self.batch_size))
    
    def __getitem__(self, idx):
        return self.load_batch(idx=idx)        



    def load_batch(self,idx=0, img_paths=None, training=True, bicubic=False):
        """ Loads a batch of images or video"""
        if(self.media_type=='i'):
            #print("1. Media type image folder")
            imgs_lr, imgs_hr = self.load_batch_image(idx, img_paths=img_paths, training=training,bicubic=False)
        elif(self.media_type=='v' and os.path.isdir(self.datapath)):
            #print("2. Media type video folder")
            imgs_lr, imgs_hr = self.load_batch_video(idx, img_paths=None, training=training, bicubic=False)
        elif(self.media_type=='v' and not os.path.isdir(self.datapath)):
            #print("3. Media type single video")
            imgs_lr, imgs_hr = self.load_batch_video(idx, img_paths=self.datapath, training=training, bicubic=False)
        elif(self.media_type=='vt' and os.path.isdir(self.datapath)):
            #print("4. Media type single video")
            imgs_lr, imgs_hr = self.load_temporal_batch_video(idx, img_paths=None, training=training, bicubic=False)
        else:
            print("Error: type of media fail")
        return imgs_lr, imgs_hr


    def load_temporal_batch_video(self, idx=0, img_paths=None, training=True, bicubic=False):
        """Loads a batch of frames from videos folder""" 
        # Starting index to look in
        cur_idx=np.random.randint(self.total_imgs)
        if not img_paths:
            cur_idx = idx*self.batch_size            
        #print('cur_idx:',cur_idx)
        if training == False:
            cur_idx = idx
            
        # Scale and pre-process images
        imgs_hr, imgs_lr = [], []
        for i in range(self.time_step):
            imgs_lr.append([])
        while True:
            # Check if done with batch        
            if img_paths is None:
                if cur_idx >= self.total_imgs:
                    cur_idx = 0
                if len(imgs_hr) >= self.batch_size:
                    break
            if img_paths is not None and len(imgs_hr) == len(img_paths):
                break
            try: 
                # Load image
                ori_imgs_hr = None
                if img_paths:
                    ori_imgs_hr = self.load_frame(img_paths[cur_idx],self.time_step)
                else:
                    ori_imgs_hr = self.load_frame(self.img_paths[cur_idx],self.time_step)

                # Create HR images to go through
                img_crops = [] 
                if training:
                    for i in range(self.crops_per_image):
                        assert ori_imgs_hr.shape[3] == self.channel
                        height, width = ori_imgs_hr.shape[1], ori_imgs_hr.shape[2]
                        x = np.random.randint(0, width - self.width_hr + 1)
                        y = np.random.randint(0, height - self.height_hr + 1)
                        t_step_img_crops = []
                        for img_hr in ori_imgs_hr:
                            t_step_img_crops.append(self.fix_crop(img_hr,self.height_hr,self.width_hr,y,x))
                        img_crops.append(t_step_img_crops)
                else:
                    assert ori_imgs_hr.shape[3] == self.channel
                    t_step_img_crops = []
                    for img_hr in ori_imgs_hr:
                       t_step_img_crops.append(img_hr)
                    img_crops.append(t_step_img_crops) 

                # Downscale the HR images and save
                for t_step_img_hr in img_crops:
                    t_step_imgs_lr = [] 
                    if img_paths is None:
                        if cur_idx >= self.total_imgs:
                            cur_idx = 0
                        if len(imgs_hr) >= self.batch_size:
                            break
                    if img_paths is not None and len(imgs_hr) == len(img_paths):
                        break 
                    i=0
                    for img_hr in t_step_img_hr:
                        # For LR, do bicubic downsampling
                        method = Image.BICUBIC if bicubic else choice(self.options)
                        lr_shape = (int(img_hr.shape[1]/self.scale), int(img_hr.shape[0]/self.scale))           
                        img_lr = Image.fromarray(img_hr.astype(np.uint8))
                        img_lr = np.array(img_lr.resize(lr_shape, method))
                        # Scale color values
                        img_hr = self.scale_hr_imgs(img_hr)
                        img_lr = self.scale_lr_imgs(img_lr)

                        # Store LR images 
                        imgs_lr[i].append(img_lr)
                        i+=1
                    # Store LR and HR images
                    imgs_hr.append(np.expand_dims(img_hr, axis=0))
            except Exception as e:
                print(e)
                pass
            finally:
                cur_idx += 1

        # Convert to numpy arrays when we are training 
        # Note: all are cropped to same size, which is not the case when not training
        for i in range(self.time_step):
            imgs_lr[i] = np.array(imgs_lr[i])
        imgs_hr = np.array(imgs_hr)
        imgs_hr=imgs_hr.reshape(self.batch_size,img_hr.shape[0],img_hr.shape[1],self.channel)

        # Return image batch
        return imgs_lr, imgs_hr



    def load_batch_video(self, idx=0, img_paths=None, training=True, bicubic=False):
        """Loads a batch of frames from videos folder""" 
        # Starting index to look in
        cur_idx = 0
        if not img_paths:
            cur_idx = idx*self.batch_size            
        #print('cur_idx:',cur_idx)
            
        # Scale and pre-process images
        imgs_hr, imgs_lr = [], []
        while True:

            # Check if done with batch
            if img_paths is None:
                if cur_idx >= self.total_imgs:
                    cur_idx = 0
                if len(imgs_hr) >= self.batch_size:
                    break
            if img_paths is not None and len(imgs_hr) == len(img_paths):
                break            
            
            try: 
                # Load image
                img_hr = None
                if img_paths:
                    #print('1. video: ',img_paths[cur_idx])
                    img_hr = self.load_frame(img_paths[cur_idx])
                    #print(img_hr.shape)
                else:
                    #print('2. video: ',self.img_paths[cur_idx])
                    img_hr = self.load_frame(self.img_paths[cur_idx])
                    #print(img_hr.shape)

                # Create HR images to go through
                img_crops = []
                if training:
                    for i in range(self.crops_per_image):
                        #print(idx, cur_idx, "Loading crop: ", i)
                        img_crops.append(self.random_crop(img_hr, (self.height_hr, self.width_hr)))
                else:
                    img_crops = [img_hr]

                # Downscale the HR images and save
                for img_hr in img_crops:

                    # TODO: Refactor this so it does not occur multiple times
                    if img_paths is None:
                        if cur_idx >= self.total_imgs:
                            cur_idx = 0
                        if len(imgs_hr) >= self.batch_size:
                            break
                    if img_paths is not None and len(imgs_hr) == len(img_paths):
                        break   

                    # For LR, do bicubic downsampling
                    method = Image.BICUBIC if bicubic else choice(self.options)
                    lr_shape = (int(img_hr.shape[1]/self.scale), int(img_hr.shape[0]/self.scale))           
                    img_lr = Image.fromarray(img_hr.astype(np.uint8))
                    img_lr = np.array(img_lr.resize(lr_shape, method))

                    # Scale color values
                    img_hr = self.scale_hr_imgs(img_hr)
                    img_lr = self.scale_lr_imgs(img_lr)

                    # Store images
                    imgs_hr.append(img_hr)
                    imgs_lr.append(img_lr)
                
            except Exception as e:
                print(e)
                pass
            finally:
                cur_idx += 1

        # Convert to numpy arrays when we are training 
        # Note: all are cropped to same size, which is not the case when not training
        if training:
            imgs_hr = np.array(imgs_hr)
            imgs_lr = np.array(imgs_lr)

        # Return image batch
        return imgs_lr, imgs_hr


    def load_batch_image(self, idx=0, img_paths=None, training=True, bicubic=False):
        """Loads a batch of images from datapath folder""" 

        # Starting index to look in
        cur_idx = 0
        
        if not img_paths:
            cur_idx = idx*self.batch_size
            
        # Scale and pre-process images
        imgs_hr, imgs_lr = [], []
        while True:

            # Check if done with batch
            if img_paths is None:
                if cur_idx >= self.total_imgs:
                    cur_idx = 0
                if len(imgs_hr) >= self.batch_size:
                    break
            if img_paths is not None and len(imgs_hr) == len(img_paths):
                break            
            
            try: 
                # Load image
                img_hr = None
                if img_paths:
                    img_hr = self.load_img(img_paths[cur_idx])
                else:
                    img_hr = self.load_img(self.img_paths[cur_idx])
                    
                # Create HR images to go through
                img_crops = []
                if training:
                    for i in range(self.crops_per_image):
                        #print(idx, cur_idx, "Loading crop: ", i)
                        img_crops.append(self.random_crop(img_hr, (self.height_hr, self.width_hr)))
                else:
                    img_crops = [img_hr]

                # Downscale the HR images and save
                for img_hr in img_crops:

                    # TODO: Refactor this so it does not occur multiple times
                    if img_paths is None:
                        if cur_idx >= self.total_imgs:
                            cur_idx = 0
                        if len(imgs_hr) >= self.batch_size:
                            break
                    if img_paths is not None and len(imgs_hr) == len(img_paths):
                        break   

                    # For LR, do bicubic downsampling
                    method = Image.BICUBIC if bicubic else choice(self.options)
                    lr_shape = (int(img_hr.shape[1]/self.scale), int(img_hr.shape[0]/self.scale))           
                    img_lr = Image.fromarray(img_hr.astype(np.uint8))
                    img_lr = np.array(img_lr.resize(lr_shape, method))

                    # Scale color values
                    img_hr = self.scale_hr_imgs(img_hr)
                    img_lr = self.scale_lr_imgs(img_lr)

                    # Store images
                    imgs_hr.append(img_hr)
                    imgs_lr.append(img_lr)
                
            except Exception as e:
                print(e)
                pass
            finally:
                cur_idx += 1

        # Convert to numpy arrays when we are training 
        # Note: all are cropped to same size, which is not the case when not training
        if training:
            imgs_hr = np.array(imgs_hr)
            imgs_lr = np.array(imgs_lr)

        # Return image batch
        return imgs_lr, imgs_hr
    
    def sr_genarator(self,model,img_lr):
        """Predict sr frame given a LR frame"""
        # Predict high-resolution version (add batch dimension to image)
        img_lr=self.scale_lr_imgs(img_lr)
        img_sr = model.generator.predict(np.expand_dims(img_lr, 0))
        # Remove batch dimension
        img_sr = img_sr.reshape(img_sr.shape[1], img_sr.shape[2], img_sr.shape[3])
        img_sr = self.unscale_hr_imgs(img_sr)
        return img_sr

    def write_sr_images(self,model, input_images, output_images):
        """        
        :param SRGAN model: The trained SRGAN model
        :param list input_images: List of filepaths for input images
        :param list output_images: List of filepaths for output images
        """
       
        # Load the images to perform test on images
        imgs_lr, imgs_hr = self.load_batch(idx=0,img_paths=input_images, training=False)
        # Scale color values
        imgs_hr = self.unscale_hr_imgs(np.array(imgs_hr))
        imgs_lr = self.unscale_lr_imgs(np.array(imgs_lr)) 

        # Create super resolution images
        imgs_sr = []
        time_elapsed = []
        for img_lr,img_hr in zip(imgs_lr,imgs_hr):
            start = timer()
            img_sr = self.sr_genarator(model,img_lr)    
            end = timer()
            time_elapsed.append(end - start)   
            
            img_sr = Image.fromarray(img_sr.astype(np.uint8))
            img_sr.save(output_images.split(".")[0]+"SR.png")
            #imageio.imwrite(output_images.split(".")[0]+"SR.png", img_sr)
            
            img_hr = Image.fromarray(img_hr.astype(np.uint8))
            img_hr.save(output_images.split(".")[0]+"HR.png")
            #imageio.imwrite(output_images.split(".")[0]+"HR.png", img_hr)
            
            img_lr = Image.fromarray(img_lr.astype(np.uint8))
            img_lr.save(output_images.split(".")[0]+"LR.png")
            #imageio.imwrite(output_images.split(".")[0]+"LR.png", img_lr)
        return time_elapsed


class VideoRestore():
    #def __init__(self):

    @staticmethod
    def scale_lr_imgs(imgs):
        """Scale low-res images prior to passing to SRGAN"""
        return imgs / 255.
        
    @staticmethod
    def unscale_hr_imgs(imgs):
        """Un-Scale high-res images"""
        return (imgs + 1.) * 127.5
    
    def count_frames_manual(self,cap):
        count=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if(ret):
                count +=1
            else:
                break
        return count
    
    def count_frames(self,cap):
        '''Count total frames in video'''
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            total = self.count_frames_manual(cap)
        return total
    
    def sr_genarator(self,model,img_lr):
        """Predict sr frame given a LR frame"""
        # Predict high-resolution version (add batch dimension to image)
        img_sr = np.squeeze(
                    model.generator.predict(img_lr,
                        batch_size=1
                    ),
                    axis=0
                )
        # Remove batch dimension
        img_sr = self.unscale_hr_imgs(img_sr)
        return img_sr
       
    def write_srvideo(self, model=None,lr_videopath=None,sr_videopath=None,print_frequency=30,crf=15):
        """Predict SR video given LR video """
        cap = cv2.VideoCapture(lr_videopath) 
        if cap.isOpened():
            fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
            # ffmpeg setup '-qscale', '5',
            p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', str(fps), '-i', '-', '-vcodec', 'libx264','-preset', 'veryslow', '-crf',str(crf), '-r', str(fps), sr_videopath], stdin=PIPE)
        else:
            print("Error to open low resolution video")
            return -1
        
        # Get video total frames
        t_frames = self.count_frames(cap)    
        #cria arquivo video hr if hr video is open
        count = 0
        time_elapsed = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                start = timer()
                img_sr = self.sr_genarator(model,frame)
                end = timer()
                time_elapsed.append(end - start)
                im = Image.fromarray(img_sr.astype(np.uint8))
                im.save(p.stdin, 'JPEG')
                count +=1
            else:
                break
            if(count % print_frequency == 0):
                print('Time per Frame: '+str(np.mean(time_elapsed))+'s')
                print('Estimated time: '+str(np.mean(time_elapsed)*(t_frames-count)/60.)+'min')
        p.stdin.close()
        p.wait()
        cap.release()
        return time_elapsed


    def write_temporal_srvideo(self, model=None,lr_videopath=None,sr_videopath=None,print_frequency=30,crf=15,time_step=1):
        """Predict SR video given LR video """
        cap = cv2.VideoCapture(lr_videopath) 
        if cap.isOpened():
            fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
            # ffmpeg setup '-qscale', '5',
            p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', str(fps), '-i', '-', '-vcodec', 'libx264','-preset', 'veryslow', '-crf',str(crf), '-r', str(fps), sr_videopath], stdin=PIPE)
        else:
            print("Error to open low resolution video")
            return -1
            
        # Get video total frames
        t_frames = self.count_frames(cap)    
        #cria arquivo video hr if hr video is open
        count = 0
        time_elapsed = []
        frame_steps = []
        cap.set(1,0)
        while cap.isOpened() and count < t_frames:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if count > 0:
                    frame_steps.append(np.expand_dims(self.scale_lr_imgs(frame),0)) 
                    frame_steps.pop(0)
                else:
                    frame_steps = [np.expand_dims(self.scale_lr_imgs(frame),0) for f in range(time_step)] 
                start = timer()
                img_sr = self.sr_genarator(model,frame_steps)
                end = timer()
                time_elapsed.append(end - start)
                im = Image.fromarray(img_sr.astype(np.uint8))
                im.save(p.stdin, 'JPEG')
                count +=1
            else:
                break
            if(count % print_frequency == 0):
                print('Time per Frame: '+str(np.mean(time_elapsed))+'s')
                print('Estimated time: '+str(np.mean(time_elapsed)*(t_frames-count)/60.)+'min')
        p.stdin.close()
        p.wait()
        cap.release()
        return time_elapsed


def plot_test_images(model, loader, datapath_test, test_output, epoch, name='SRGAN'):
    """        
    :param SRGAN model: The trained SRGAN model
    :param DataLoader loader: Instance of DataLoader for loading images
    :param str datapath_test: path to folder with testing images
    :param string test_output: Directory path for outputting testing images
    :param int epoch: Identifier for how long the model has been trained
    """

    try:   
        # Get the location of test images
        test_images = [os.path.join(datapath_test, f) for f in os.listdir(datapath_test) if any(filetype in f.lower() for filetype in ['jpeg','mp4','264', 'png', 'jpg'])]
        
        # Load the images to perform test on images
        imgs_lr, imgs_hr = loader.load_batch(img_paths=test_images, training=False, bicubic=True)

        # Create super resolution and bicubic interpolation images
        imgs_sr = []
        imgs_bc = []
        for i in range(len(test_images)):

            # Bicubic interpolation
            pil_img = loader.unscale_lr_imgs(imgs_lr[i]).astype('uint8')
            pil_img = Image.fromarray(pil_img)        
            hr_shape = (imgs_hr[i].shape[1], imgs_hr[i].shape[0])

            imgs_bc.append(
                loader.scale_lr_imgs(
                    np.array(pil_img.resize(hr_shape, resample=Image.BICUBIC))
                )
            )

            # SRGAN prediction
            imgs_sr.append(
                np.squeeze(
                    model.predict(
                        np.expand_dims(imgs_lr[i], 0),
                        batch_size=1
                    ),
                    axis=0
                )
            )

        # Unscale colors values
        imgs_lr = [loader.unscale_lr_imgs(img).astype(np.uint8) for img in imgs_lr]
        imgs_hr = [loader.unscale_hr_imgs(img).astype(np.uint8) for img in imgs_hr]
        imgs_sr = [loader.unscale_hr_imgs(img).astype(np.uint8) for img in imgs_sr]
	

        # Loop through images
        for img_hr, img_lr, img_bc, img_sr, img_path in zip(imgs_hr, imgs_lr, imgs_bc, imgs_sr, test_images):

            # Get the filename
            filename = os.path.basename(img_path).split(".")[0]
	    
	    
            # Images and titles
            images = {
                'Low Resolution': img_lr, 
                'Bicubic Interpolation': img_bc, 
                name: img_sr, 
                'Original': img_hr
            }
	    

        # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1                    
            fig, axes = plt.subplots(1, 4, figsize=(40, 10))
            for i, (title, img) in enumerate(images.items()):
                axes[i].imshow(img)
                axes[i].set_title("{} - {}".format(title, img.shape))
                axes[i].axis('off')
            plt.suptitle('{} - Epoch: {}'.format(filename, epoch))

            # Save directory                    
            savefile = os.path.join(test_output, "{}-Epoch{}.png".format(filename, epoch))
            fig.savefig(savefile)
            plt.close()
            gc.collect()
    except Exception as e:
        print(e)


def plot_temporal_test_images(model, loader, datapath_test, test_output, epoch, name='TSRGAN',time_step=3):
    """        
    :param SRGAN model: The trained SRGAN model
    :param DataLoader loader: Instance of DataLoader for loading images
    :param str datapath_test: path to folder with testing images
    :param string test_output: Directory path for outputting testing images
    :param int epoch: Identifier for how long the model has been trained
    """

    try:   
        # Get the location of test images
        test_images = [os.path.join(datapath_test, f) for f in os.listdir(datapath_test) if any(filetype in f.lower() for filetype in ['mp4','webm','264'])]
       
        # Create super resolution and bicubic interpolation images
        imgs_sr = []
        imgs_bc = []
        _imgs_lr = []
        _imgs_hr = []
        for i in range(len(test_images)):
            # Load the images to perform test on images
            imgs_lr, imgs_hr = loader.load_batch(idx=i,training=True, bicubic=True)
            

            # Bicubic interpolation
            pil_img = loader.unscale_lr_imgs(np.squeeze(imgs_lr[time_step-1], axis=0)).astype('uint8')
            pil_img = Image.fromarray(pil_img)        
            hr_shape = (imgs_hr.shape[2], imgs_hr.shape[1])

            _imgs_lr.append(np.squeeze(imgs_lr[time_step-1], axis=0))
            _imgs_hr.append(np.squeeze(imgs_hr, axis=0))
            imgs_bc.append(
                loader.scale_lr_imgs(
                    np.array(pil_img.resize(hr_shape, resample=Image.BICUBIC))
                )
            )

            # TSRGAN prediction
            imgs_sr.append(
                np.squeeze(
                    model.generator.predict(imgs_lr,
                        batch_size=1
                    ),
                    axis=0
                )
            )

        # Unscale colors values
        imgs_lr = [loader.unscale_lr_imgs(img).astype(np.uint8) for img in _imgs_lr]
        imgs_hr = [loader.unscale_hr_imgs(img).astype(np.uint8) for img in _imgs_hr]
        imgs_sr = [loader.unscale_hr_imgs(img).astype(np.uint8) for img in imgs_sr]
	

        # Loop through images
        for img_hr, img_lr, img_bc, img_sr, img_path in zip(imgs_hr, imgs_lr, imgs_bc, imgs_sr, test_images):

            # Get the filename
            filename = os.path.basename(img_path).split(".")[0]
	    
	    
            # Images and titles
            images = {
                'Low Resolution': [img_lr, img_hr],
                'Bicubic Interpolation': [img_bc,img_hr], 
                name: [img_sr,img_hr], 
                'Original': [img_hr,None]
            }
	    
            # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1                    
            fig, axes = plt.subplots(1, 4, figsize=(40, 10))
            for i, (title, imgs) in enumerate(images.items()):
                axes[i].imshow(imgs[0])
                axes[i].set_title("{} - {} {}".format(title, imgs[0].shape, ("- psnr: "+str(round(psnr(imgs[1],imgs[0]),2)) if (title == name or title == 'Bicubic Interpolation') else " ")))
                axes[i].axis('off')
            plt.suptitle('{} - Epoch: {}'.format(filename, epoch))

            # Save directory                    
            savefile = os.path.join(test_output, "{}-Epoch{}.png".format(filename, epoch))
            fig.savefig(savefile)
            plt.close()
            gc.collect()
    except Exception as e:
        print(e)


def PSNR(y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        
        Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
        """
        return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


