# 1. TPU 접속 경로 설정.
echo 'export XRT_TPU_CONFIG="localservice;0;localhost:51011"' >> ~/.bashrc  # bash를 사용하는 경우
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

# 2. 기본 Python 버전 변경.
# sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 2
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 9
# sudo update-alternatives --config python

# 3. pip 버전 업그레이드.
pip install --upgrade pip

pip install jax[tpu]==0.4.3 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax==0.6.5
pip install orbax==0.1.1
pip install tabulate
pip install tqdm
pip install wandb==0.13.10
pip install git+https://github.com/cs-giung/giung2-dev.git@acbf4f7f4ce62ea882d3d88a345b1b2a40396b02
pip install easydict
pip install einops
pip install scikit-learn
pip install seaborn

cd ~/dbn
mkdir checkpoints

cd ~/dbn/data
wget https://www.dropbox.com/s/urh77zy42xxzhgo/ImageNet1k_x64.tar.gz
tar -xvzf ImageNet1k_x64.tar.gz
cd ~/dbn