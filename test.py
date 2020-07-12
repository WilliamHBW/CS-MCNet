from config import opt 
import torch as t 
import numpy as np 
import os 
from models.MCNet import MCNet
import utils 
import math 
from skimage.measure import compare_psnr 
from skimage.measure import compare_ssim
import time
import torchsnooper
import cv2


@t.no_grad()
def test(**kwargs):
    #refresh parameter
    opt._parse(kwargs)

    #create save folder, open log file
    save_test_root = opt.save_test_root
    if not os.path.exists(save_test_root):
        os.makedirs(save_test_root)
    log_file = open(save_test_root+"/result.txt",mode='w')

    #set gpu environment, load weights
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    bernoulli_weights = np.loadtxt(opt.weights)
    bernoulli_weights = t.from_numpy(bernoulli_weights).float().to(opt.device)

    #initialize model, load model file
    model = MCNet(bernoulli_weights,opt.cr,opt.blk_size,opt.ref_size).eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device) 
    model.eval()

    #get test videos
    videos = [os.path.join(opt.test_data_root,video) for video in os.listdir(opt.test_data_root)]
    video_num = len(videos)
    print("total test video number:",video_num)

    end = time.time()
    psnr_av = 0
    ssim_av = 0
    time_av = 0
    for item in videos:
        if (item.split(".")[-1]!='avi'):
            continue
        print("now is processing:",item)
        log_file.write("%s"%item)
        log_file.write("\n")
        
        uv = utils.Video(opt.height,opt.width)
        test_data = uv.video2array(item,opt.frame_num)
        test_data_t = t.from_numpy(test_data).float().to(opt.device)
        result_data_t = t.zeros_like(test_data_t).cuda()

        psnr_total = 0
        ssim_total = 0
        frame_cnt = 0
        
        #do test on every video
        for i in range(test_data_t.size(0)):
            for j in range(test_data_t.size(1)):
                
                frames = test_data_t[i,j,:,:,:]
                frames_num = frames.size(0)
        
                result_frame = t.ones(1,frames[0].size(0),frames[0].size(1)).float().to(opt.device)
                result_frames = t.zeros(frames_num,frames[0].size(0),frames[0].size(1)).to(opt.device)
                frames_t = frames

                x_b = uv.frame_unfold(frames_t,opt.blk_size,int(opt.blk_size/2)).to(opt.device)
                blk_num_h = x_b.size(1)
                blk_num_w = x_b.size(2)
        
                for ii in range(frames_num):
                    x_ref_b = uv.frame_unfold(result_frame,opt.ref_size,int(opt.ref_size/2))
                    result_b = t.zeros_like(x_b[0].unsqueeze_(0))

                    input_ = (x_b[ii,:,:,:,:]/255.0).float().to(opt.device)
                    input_target = input_.view(1*blk_num_h*blk_num_w,opt.blk_size,opt.blk_size)
                    input_m = input_.view(1*blk_num_h*blk_num_w,opt.blk_size*opt.blk_size,1)

                    ref = x_ref_b.repeat(1,2,1,1,1)
                    ref[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],:,:,:] = ref[:,[0,9,1,10,2,11,3,12,4,13,5,14,6,15,7,16,8,17],:,:,:]
                    ref = ref.repeat(1,1,2,1,1)
                    ref[:,:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],:,:] = ref[:,:,[0,9,1,10,2,11,3,12,4,13,5,14,6,15,7,16,8,17],:,:]
                    ref_cat = ref[:,:,-1,:,:].view(1,ref.size(1),1,opt.ref_size,opt.ref_size)
                    ref = t.cat((ref,ref_cat),2)
                    ref_cat = ref[:,-1,:,:,:].view(1,1,ref.size(2),opt.ref_size,opt.ref_size)
                    ref = t.cat((ref,ref_cat),1)
                    ref = ref.view(1*blk_num_h*blk_num_w,opt.ref_size,opt.ref_size)

                    b_s = input_target.size(0)
                    weight = bernoulli_weights.unsqueeze(0).repeat(b_s,1,1).to(opt.device)
                    input = t.bmm(weight,input_m).squeeze_(2)
                    if(opt.noise_snr>0):
                        input = add_noise(input,opt.noise_snr,10)

                    output,_ = model(input,ref,input)
                    result_b = output.view(1,blk_num_h,blk_num_w,opt.blk_size,opt.blk_size)

                    frame_cnt = frame_cnt + 1
                    result_frame = uv.frame_fold(result_b,opt.blk_size,int(opt.blk_size/2))
                 
                    result_frames[ii] = result_frame
                    psnr = compare_psnr((frames_t[ii].unsqueeze(0)).cpu().numpy(),(result_frame*255).cpu().numpy(),data_range=255)
                    ssim = compare_ssim(frames_t[ii].cpu().numpy(),(result_frame*255).squeeze(0).cpu().numpy())
                    psnr_total = psnr_total + psnr
                    ssim_total = ssim_total + ssim
                
                result_data_t[i,j,:,:,:] = result_frames

        uv.array2video(result_data_t,opt.save_test_root)

        #get log information
        video_time = time.time() - end
        info = str(psnr_total/frame_cnt)+"\n"
        log_file.write("%s"%info)
        info = str(ssim_total/frame_cnt)+"\n"
        log_file.write("%s"%info)
        info = str(video_time/frame_cnt)+"\n"
        log_file.write("%s"%info)
        end = time.time()

        print("PSNR is:",psnr_total/frame_cnt,"SSIM is:",ssim_total/frame_cnt,"Time per frame is:",video_time/frame_cnt)
        psnr_av = psnr_av + psnr_total/frame_cnt
        ssim_av = ssim_av + ssim_total/frame_cnt
        time_av = time_av + video_time/frame_cnt
    log_file.close()

    print("Average PSNR is:",psnr_av/video_num,"Average SSIM is:",ssim_av/video_num,"Average Time per frame is:",time_av/video_num)

def add_noise(input,SNR,seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True

    input_np = input.cpu().numpy()
    noise = np.random.randn(input_np.shape[0],input_np.shape[1]).astype(np.float32)
    noise = noise - np.mean(noise)
    signal_power = np.linalg.norm(input_np)**2/(input_np.size)
    noise_power = signal_power/np.power(10,(SNR/10))
    noise = (np.sqrt(noise_power)/np.std(noise))*noise

    y = input_np + noise 
    y = t.from_numpy(y).cuda()
    return y

if __name__=='__main__':
    import fire
    fire.Fire()






            