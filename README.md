# CS-MCNet
Pytorch code of CS-MCNet

Test usage: put test video in ./data/test.

> python test.py test --height=720 --width=1280 --load_model_path='./checkpoints/MCNet_4_cr16/best_model.pth' --frame_num=180

default test result is in ./results and other parameters can be find in config.py.

Train usage: download our train dataset and put it in ./data/train. 
           
> python train.py train --save_train_root='./checkpoints/default_train' --cr=1/16 --weights='./weights/weights_cr16.txt' --batch_size=4 --max_epoch=50 --print_freq=10 --lr=0.01

default train result is in ./checkpoints


