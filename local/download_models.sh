#!/bin/bash
LOC=checkpoints
mkdir -p "$LOC"

# Which models to download?
IMAGENET_1024=false
IMAGENET_16384=true
GUMBEL=false
#WIKIART_1024=false
WIKIART_16384=false
COCO=false
FACESHQ=false
SFLCKR=false


if [ "$IMAGENET_1024" = true ] ; then
  # imagenet_1024 - 958 MB:
  if [ ! -f "$LOC"/vqgan_imagenet_f16_1024.yaml ]; then
    curl -L -o "$LOC"/vqgan_imagenet_f16_1024.yaml -C - 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 1024
  fi
  if [ ! -f "$LOC"/vqgan_imagenet_f16_1024.ckpt ]; then
    curl -L -o "$LOC"/vqgan_imagenet_f16_1024.ckpt -C - 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 1024
  fi
fi

if [ "$IMAGENET_16384" = true ] ; then
  # imagenet_16384 - 980 MB:
  if [ ! -f "$LOC"/vqgan_imagenet_f16_16384.yaml ]; then
    curl -L -o "$LOC"/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
  fi
  if [ ! -f "$LOC"/vqgan_imagenet_f16_16384.ckpt ]; then
    curl -L -o "$LOC"/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
  fi
fi

if [ "$GUMBEL" = true ] ; then
  # vqgan_gumbel_f8_8192 (was openimages_f16_8192) - 376 MB:
  if [ ! -f "$LOC"/vqgan_gumbel_f8_8192.yaml ]; then
    curl -L -o "$LOC"/vqgan_gumbel_f8_8192.yaml -C - 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' 
  fi
  if [ ! -f "$LOC"/vqgan_gumbel_f8_8192.ckpt ]; then
    curl -L -o "$LOC"/vqgan_gumbel_f8_8192.ckpt -C - 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1'    
  fi
fi

if [ "$COCO" = true ] ; then
  # coco - 8.4 GB:
  if [ ! -f "$LOC"/coco.yaml ]; then
    curl -L -o "$LOC"/coco.yaml -C - 'https://dl.nmkd.de/ai/clip/coco/coco.yaml' #COCO
  fi
  if [ ! -f "$LOC"/coco.ckpt ]; then
    curl -L -o "$LOC"/coco.ckpt -C - 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt' #COCO
  fi
fi
  
if [ "$FACESHQ" = true ] ; then
# faceshq:
  if [ ! -f "$LOC"/faceshq.yaml ]; then
    curl -L -o "$LOC"/faceshq.yaml -C - 'https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT' #FacesHQ
  fi
  if [ ! -f "$LOC"/faceshq.ckpt ]; then
    curl -L -o "$LOC"/faceshq.ckpt -C - 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt' #FacesHQ
  fi
fi

# Link?
#if [ "$WIKIART_1024" = true ] ; then   
#  # wikiart_1024 - 958 MB: 
#  if [ ! -f "$LOC"/wikiart_1024.yaml ]; then
#    curl -L -o "$LOC"/wikiart_1024.yaml -C - 'http://mirror.io.community/blob/vqgan/wikiart.yaml' #WikiArt 1024
#  fi
#  if [ ! -f "$LOC"/wikiart_1024.ckpt ]; then
#    curl -L -o "$LOC"/wikiart_1024.ckpt -C - 'http://mirror.io.community/blob/vqgan/wikiart.ckpt' #WikiArt 1024
#  fi
#fi

if [ "$WIKIART_16384" = true ] ; then
  #wikiart_16384 - 1 GB: 
  if [ ! -f "$LOC"/wikiart_16384.yaml ]; then
    curl -L -o "$LOC"/wikiart_16384.yaml -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml' #WikiArt 16384
  fi
  if [ ! -f "$LOC"/wikiart_16384.ckpt ]; then
    curl -L -o "$LOC"/wikiart_16384.ckpt -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt' #WikiArt 16384
  fi
fi

if [ "$SFLCKR" = true ] ; then
# sflckr:
  if [ ! -f "$LOC"/sflckr.yaml ]; then
    curl -L -o "$LOC"/sflckr.yaml -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1' #S-FLCKR
  fi
  if [ ! -f "$LOC"/sflckr.ckpt ]; then
    curl -L -o "$LOC"/sflckr.ckpt -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1' #S-FLCKR
  fi
fi

# Others:

#ade20k:
#  curl -L -o ade20k.yaml -C - 'https://static.miraheze.org/intercriaturaswiki/b/bf/Ade20k.txt' #ADE20K
#  curl -L -o ade20k.ckpt -C - 'https://app.koofr.net/content/links/0f65c2cd-7102-4550-a2bd-07fd383aac9e/files/get/last.ckpt?path=%2F2020-11-20T21-45-44_ade20k_transformer%2Fcheckpoints%2Flast.ckpt' #ADE20K
  
#ffhq:
#  curl -L -o ffhq.yaml -C - 'https://app.koofr.net/content/links/0fc005bf-3dca-4079-9d40-cdf38d42cd7a/files/get/2021-04-23T18-19-01-project.yaml?path=%2F2021-04-23T18-19-01_ffhq_transformer%2Fconfigs%2F2021-04-23T18-19-01-project.yaml&force' #FFHQ
#  curl -L -o ffhq.ckpt -C - 'https://app.koofr.net/content/links/0fc005bf-3dca-4079-9d40-cdf38d42cd7a/files/get/last.ckpt?path=%2F2021-04-23T18-19-01_ffhq_transformer%2Fcheckpoints%2Flast.ckpt&force' #FFHQ
  
#celebahq:
#  curl -L -o celebahq.yaml -C - 'https://app.koofr.net/content/links/6dddf083-40c8-470a-9360-a9dab2a94e96/files/get/2021-04-23T18-11-19-project.yaml?path=%2F2021-04-23T18-11-19_celebahq_transformer%2Fconfigs%2F2021-04-23T18-11-19-project.yaml&force' #CelebA-HQ
#  curl -L -o celebahq.ckpt -C - 'https://app.koofr.net/content/links/6dddf083-40c8-470a-9360-a9dab2a94e96/files/get/last.ckpt?path=%2F2021-04-23T18-11-19_celebahq_transformer%2Fcheckpoints%2Flast.ckpt&force' #CelebA-HQ

#
