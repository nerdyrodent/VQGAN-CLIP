# VQGAN-CLIP
Just me playing with getting VQGAN+CLIP running locally, rather than having to use colab.
Taken from the following google colab - https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ

Tested on Ubuntu 20.04, Nvidia RTX 3090.

Still a work in progress

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
# Run
```
python generate.py
```

## Multiple prompts

Text and image prompts can be split using the pipe symbol, "|". For example:

```
python generate.py -p "A picture of a snowflake | zoom | high definition"
```

## Advanced options
```
generate.py [-h] [-p PROMPTS] [-o OUTPUT] [-i MAX_ITERATIONS] [-ip IMAGE_PROMPTS]
[-nps NOISE_PROMPT_SEEDS] [-npw NOISE_PROMPT_WEIGHTS] [-s SIZE SIZE] [-ii INIT_IMAGE]
[-iw INIT_WEIGHT] [-m CLIP_MODEL] [-conf VQGAN_CONFIG] [-ckpt VQGAN_CHECKPOINT]
[-lr STEP_SIZE] [-cuts CUTN] [-cutp CUT_POW] [-se DISPLAY_FREQ] [-sd SEED]
```

```
optional arguments:
  -h, --help            show this help message and exit
  -p PROMPTS, --prompts PROMPTS
                        Text prompts
  -o OUTPUT, --output OUTPUT
                        Number of iterations
  -i MAX_ITERATIONS, --iterations MAX_ITERATIONS
                        Number of iterations
  -ip IMAGE_PROMPTS, --image_prompts IMAGE_PROMPTS
                        Image prompts / target image
  -nps NOISE_PROMPT_SEEDS, --noise_prompt_seeds NOISE_PROMPT_SEEDS
                        Noise prompt seeds
  -npw NOISE_PROMPT_WEIGHTS, --noise_prompt_weights NOISE_PROMPT_WEIGHTS
                        Noise prompt seeds
  -s SIZE SIZE, --size SIZE SIZE
                        Image size (width, height)
  -ii INIT_IMAGE, --init_image INIT_IMAGE
                        Initial image
  -iw INIT_WEIGHT, --init_weight INIT_WEIGHT
                        Initial image weight
  -m CLIP_MODEL, --clip_model CLIP_MODEL
                        CLIP model
  -conf VQGAN_CONFIG, --vqgan_config VQGAN_CONFIG
                        VQGAN config
  -ckpt VQGAN_CHECKPOINT, --vqgan_checkpoint VQGAN_CHECKPOINT
                        VQGAN checkpoint
  -lr STEP_SIZE, --learning_rate STEP_SIZE
                        Learning rate
  -cuts CUTN, --num_cuts CUTN
                        Number of cuts
  -cutp CUT_POW, --cut_power CUT_POW
                        Cut power
  -se DISPLAY_FREQ, --save_every DISPLAY_FREQ
                        Save image iterations
  -sd SEED, --seed SEED
                        Save image iterations
```
