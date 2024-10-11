# yolov5-PCB-Detection


http://www.jinglingbiaozhu.com/


```python

(yolov5) wangwei83@wangwei83-System-Product-Name:~/Desktop/yolov5-PCB-Detection$ nvidia-smi
Fri Oct 11 23:30:08 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.256.02   Driver Version: 470.256.02   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   60C    P8    25W / 350W |   5798MiB / 12053MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1079      G   /usr/lib/xorg/Xorg                  4MiB |
|    0   N/A  N/A      1770      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
(yolov5) wangwei83@wangwei83-System-Product-Name:~/Desktop/yolov5-PCB-Detection$ code
(yolov5) wangwei83@wangwei83-System-Product-Name:~/Desktop/yolov5-PCB-Detection$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_Mar__8_18:18:20_PST_2022
Cuda compilation tools, release 11.6, V11.6.124
Build cuda_11.6.r11.6/compiler.31057947_0


(yolov5) wangwei83@wangwei83-System-Product-Name:~/Desktop/yolov5-PCB-Detection$ nvidia-smi
Fri Oct 11 23:30:08 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.256.02   Driver Version: 470.256.02   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   60C    P8    25W / 350W |   5798MiB / 12053MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1079      G   /usr/lib/xorg/Xorg                  4MiB |
|    0   N/A  N/A      1770      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
(yolov5) wangwei83@wangwei83-System-Product-Name:~/Desktop/yolov5-PCB-Detection$ 






conda create -n yolov5

conda activate yolov5

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia



pip install -r requirements.txt

sudo mount -t cifs //192.168.3.130/wangwei83nas_disk1 /mnt/smbmount -o username=wangwei83nas,password=kaiyuan1028,vers=1.0,iocharset=utf8,dir_mode=0777,file_mode=0777

python train.py --img 640 --epochs 300 --data ./data/PCB.yaml