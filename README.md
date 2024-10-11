# yolov5-PCB-Detection


http://www.jinglingbiaozhu.com/


conda create -n yolov5

conda activate yolov5

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia



pip install -r requirements.txt

sudo mount -t cifs //192.168.3.130/wangwei83nas_disk1 /mnt/smbmount -o username=wangwei83nas,password=kaiyuan1028,vers=1.0,iocharset=utf8,dir_mode=0777,file_mode=0777

python train.py --img 640 --epochs 300 --data ./data/PCB.yaml