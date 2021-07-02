# VQGAN-CLIP
Just me playing with getting VQGAN + CLIP running locally, rather than having to use colab.
Taken from the following google colab - https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ

# Install
```
conda create --name vqgan python=3.9
conda activate vqgan

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/openai/CLIP
pip install taming-transformers

git clone https://github.com/CompVis/taming-transformers.git
pip install ftfy regex tqdm omegaconf pytorch-lightning IPython

pip install kornia imageio imageio-ffmpeg einops 
```
