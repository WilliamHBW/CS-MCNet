from config import opt 
import torch as t 
import os 
from models.MCNet import MCNet
from data.dataset import MCData
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm 
import math 
import numpy as np 
import utils
import matplotlib.pyplot as plt 
import torchsnooper

def train(**kwargs):
    opt._parse(kwargs)
    vis = utils.Visualizer(opt.env,port=opt.vis_port)

    save_train_root = opt.save_train_root
    if not os.path.exists(save_train_root):
        os.makedirs(save_train_root)
    opt.write_config(kwargs,opt.save_train_root)
    log_file = open(save_train_root+"/log_file.txt",mode="w")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    bernoulli_weights = np.loadtxt(opt.weights)
    bernoulli_weights = t.from_numpy(bernoulli_weights).float().to(opt.device)
    print("load weights successfully!")

    model = MCNet(bernoulli_weights,opt.cr,opt.blk_size,opt.ref_size)
    if opt.load_model_path:
        model.load(opt.load_model_path)
        print("model is loaded!")
    if opt.pre_load_model_path:
        model.block1.pre_load(opt.pre_load_model_path)
        model.block2.pre_load(opt.pre_load_model_path)
        model.block3.pre_load(opt.pre_load_model_path)
        model.block4.pre_load(opt.pre_load_model_path)
        print("pre-trained is loaded!")
    model.to(opt.device)

    train_data = MCData(opt.train_data_root,train=True)
    val_data = MCData(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,shuffle=False,num_workers=opt.num_workers)

    criterion = t.nn.MSELoss()
    lr = opt.lr 
    optimizer = t.optim.SGD(model.parameters(),lr=lr,momentum=0.9)

    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10

    loss_list = []
    loss_list_val = []

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        for ii,data in tqdm(enumerate(train_dataloader)):
            img_data_b = data[0].to(opt.device).float()
            img_data_b.squeeze_(1)
            ref_data_b = data[1].to(opt.device).float()
            ref_data_b.squeeze_(1)
            b_s = img_data_b.size(0)*img_data_b.size(1)*img_data_b.size(2)
            input_ = img_data_b.view(b_s,opt.blk_size*opt.blk_size,1)
            target = img_data_b.view(b_s,opt.blk_size,opt.blk_size)
            ref = ref_data_b.view(b_s,opt.ref_size,opt.ref_size)
            weight = bernoulli_weights.unsqueeze(0).repeat(b_s,1,1).to(opt.device)
            input = t.bmm(weight,input_).squeeze_(2)

            optimizer.zero_grad()
            output,output_mc = model(input,ref,input)

            loss = criterion(output,target) + opt.alpha*criterion(output_mc,target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
           
        if(epoch%opt.print_freq==0):
            vis.plot('loss','train_loss',loss_meter.value()[0])
            loss_list.append(loss_meter.value()[0])
            model.save(name = save_train_root+"/"+"checkpoints.pth")
            val(model,val_dataloader,loss_list_val,vis)
            info = str(epoch) + " epoch train loss value is:" + str(loss_meter.value()[0]) + "lr is: " + str(lr) +"\n"
            print(info)
            vis.log("epoch:{epoch},loss:{loss}".format(epoch=epoch,loss=loss_meter.value()[0]))
            log_file.write("%s"%info)
            
        if loss_meter.value()[0] < previous_loss:
            model.save(name = save_train_root+"/"+'best_model.pth')
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]
        
    model.save(name = save_train_root+"/"+"checkpoints.pth")
    log_file.close()
    plt.grid()
    x_axi = np.arange(len(loss_list))
    y_axi = np.array(loss_list)
    x_axi_val = np.arange(len(loss_list_val))
    y_axi_val = np.array(loss_list_val)
    plt.plot(x_axi,y_axi,'b-',label='train_loss')
    plt.plot(x_axi_val,y_axi_val,'y-',label='val_loss')
    plt.legend()
    plt.savefig(save_train_root + "/" + "train_loss.png")

@t.no_grad()
def val(model,dataloader,loss_list,vis):
    model.eval()
    
    criterion = t.nn.MSELoss()
    loss_meter = meter.AverageValueMeter()

    loss_meter.reset()

    for ii,data in tqdm(enumerate(dataloader)):
        img_data_b = data[0].to(opt.device).float()
        img_data_b.squeeze_(1)
        ref_data_b = data[1].to(opt.device).float()
        ref_data_b.squeeze_(1)
        b_s = img_data_b.size(0)*img_data_b.size(1)*img_data_b.size(2)
        input_ = img_data_b.view(b_s,opt.blk_size*opt.blk_size,1)
        target = img_data_b.view(b_s,opt.blk_size,opt.blk_size)
        ref = ref_data_b.view(b_s,opt.ref_size,opt.ref_size)
        weight = bernoulli_weights.unsqueeze(0).repeat(b_s,1,1).to(opt.device)
        input = t.bmm(weight,input_).squeeze_(2)
        
        output,output_mc = model(input,ref,input)
        loss = criterion(output,target) + opt.alpha*criterion(output_mc,target)
        loss_meter.add(loss.item())
    loss_list.append(loss_meter.value()[0])
    vis.plot('loss','val_loss',loss_meter.value()[0])
    model.train()
    
if __name__=='__main__':
    import fire
    fire.Fire()
