# coding:utf8
import warnings


class DefaultConfig(object):
    #visualization parameter
    env = 'default'  # visdom environment
    vis_port =8097 # visdom port


    #load file parameter
    train_data_root = './data/train'
    test_data_root = './data/test'
    load_model_path = None
    pre_load_model_path = None
    save_test_root = './results'
    save_train_root = './checkpoints'
    weights = './weights/weights_cr16.txt'

    #training parameter
    batch_size = 10  # batch size
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch
    max_epoch = 10
    lr = 0.001 # initial learning rate
    momentum = 0.9
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    lr_decay_ever = 3
    weight_decay = 0  # 损失函数

    #test related parameter
    frame_num = 32

    #model related parameter
    cr = 1/16
    height = 160
    width = 160
    blk_size = 16
    ref_size = 32
    alpha = 0.5
    noise_snr = 0

    device = 'cuda'

    #refresh config
    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))
    
    #save config when training
    def write_config(self,kwargs,save_root):
        f = open(save_root+"/"+"config.txt","w")
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                #print(k, getattr(self, k))
                config_info = k + str(getattr(self,k))
                f.write("%s"%config_info)
                f.write("\n")
        f.close()


opt = DefaultConfig()
