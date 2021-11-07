"""
clone the following repo if haven't
- git clone 'https://github.com/openai/CLIP'
- git clone 'https://github.com/CompVis/taming-transformers'
"""

import sys
import tempfile
import warnings
import numpy as np
from pathlib import Path
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
from omegaconf import OmegaConf
from torch_optimizer import DiffGrad, AdamP, RAdam
import kornia.augmentation as K
import imageio
from tqdm import tqdm
import cog
from CLIP import clip
from PIL import ImageFile, Image, PngImagePlugin, ImageChops

sys.path.append("taming-transformers")
from taming.models import cond_transformer, vqgan

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        (input,) = ctx.saved_tensors
        return (
            grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
            None,
            None,
        )


replace_grad = ReplaceGrad.apply
clamp_with_grad = ClampWithGrad.apply


class Predictor(cog.Predictor):
    def setup(self):
        self.device = torch.device("cuda:0")
        # Check for GPU and reduce the default image size if low VRAM
        default_image_size = 512  # >8GB VRAM
        if not torch.cuda.is_available():
            default_image_size = 256  # no GPU found
        elif (
            get_device_properties(0).total_memory <= 2 ** 33
        ):  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
            default_image_size = 318  # <8GB VRAM

        self.args = get_args()
        self.args.size = [default_image_size, default_image_size]
        self.model = load_vqgan_model(
            self.args.vqgan_config, self.args.vqgan_checkpoint
        ).to(self.device)
        print("Model loaded!")
        jit = True if float(torch.__version__[:3]) < 1.8 else False
        self.perceptor = (
            clip.load(self.args.clip_model, jit=jit)[0]
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )
        cut_size = self.perceptor.visual.input_resolution
        # choose latest Cutout class as default
        self.make_cutouts = MakeCutouts(
            cut_size, self.args.cutn, self.args, cut_pow=self.args.cut_pow
        )

        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[
            None, :, None, None
        ]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[
            None, :, None, None
        ]

        print("Using device:", self.device)
        print("Optimising using:", self.args.optimiser)

    @cog.input(
        "image",
        type=Path,
        default=None,
        help="Initial Image, optional. When the image is provided, the prompts will be used to create some 'style transfer' effect",
    )
    @cog.input(
        "prompts",
        type=str,
        default="A cute, smiling, Nerdy Rodent",
        help="Prompts for generating images. Supports multiple prompts separated by pipe | ",
    )
    @cog.input(
        "iterations",
        type=int,
        default=300,
        help="total iterations for generating images. Set to lower iterations when initial image is uploaded",
    )
    @cog.input(
        "display_frequency",
        type=int,
        default=20,
        help="display frequency for intermediate generated images",
    )
    def predict(self, image, prompts, iterations, display_frequency):
        # gumbel is False
        e_dim = self.model.quantize.e_dim
        n_toks = self.model.quantize.n_e
        f = 2 ** (self.model.decoder.num_resolutions - 1)
        toksX, toksY = self.args.size[0] // f, self.args.size[1] // f
        sideX, sideY = toksX * f, toksY * f

        if image is not None:
            self.args.init_image = str(image)
            self.args.step_size = 0.25
            if "http" in self.args.init_image:
                img = Image.open(urlopen(self.args.init_image))
            else:
                img = Image.open(self.args.init_image)
            pil_image = img.convert("RGB")
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            z, *_ = self.model.encode(pil_tensor.to(self.device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(
                torch.randint(n_toks, [toksY * toksX], device=self.device), n_toks
            ).float()
            # gumbel is False
            z = one_hot @ self.model.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

        z_orig = z.clone()
        z.requires_grad_(True)

        self.opt = get_opt(self.args.optimiser, self.args.step_size, z)

        self.args.display_freq = display_frequency
        self.args.max_iterations = iterations

        story_phrases = [phrase.strip() for phrase in prompts.split("^")]

        # Make a list of all phrases
        all_phrases = []
        for phrase in story_phrases:
            all_phrases.append(phrase.split("|"))

        # First phrase
        prompts = all_phrases[0]

        pMs = []
        for prompt in prompts:
            txt, weight, stop = split_prompt(prompt)
            embed = self.perceptor.encode_text(
                clip.tokenize(txt).to(self.device)
            ).float()
            pMs.append(Prompt(embed, weight, stop).to(self.device))
        # args.image_prompts is None for now
        # args.noise_prompt_seeds, args.noise_prompt_weights None for now
        print(f"Using text prompts: {prompts}")
        if self.args.init_image:
            print(f"Using initial image: {self.args.init_image}")

        if self.args.seed is None:
            seed = torch.seed()
        else:
            seed = self.args.seed
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")
        i = 0  # Iteration counter
        # j = 0  # Zoom video frame counter
        # p = 1  # Phrase counter
        # smoother = 0  # Smoother counter
        # this_video_frame = 0  # for video styling

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        # Do it
        for i in range(1, self.args.max_iterations + 1):
            self.opt.zero_grad(set_to_none=True)
            lossAll = ascend_txt(
                i, z, self.perceptor, self.args, self.model, self.make_cutouts, pMs
            )

            if i % self.args.display_freq == 0 and not i == self.args.max_iterations:
                yield checkin(i, lossAll, prompts, self.model, z, out_path)

            loss = sum(lossAll)
            loss.backward()
            self.opt.step()

            # with torch.no_grad():
            with torch.inference_mode():
                z.copy_(z.maximum(self.z_min).minimum(self.z_max))

            # Ready to stop yet?
            if i == self.args.max_iterations:
                yield checkin(i, lossAll, prompts, self.model, z, out_path)


@torch.inference_mode()
def checkin(i, losses, prompts, model, z, outpath):
    losses_str = ", ".join(f"{loss.item():g}" for loss in losses)
    tqdm.write(f"i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}")
    out = synth(z, model)
    info = PngImagePlugin.PngInfo()
    info.add_text("comment", f"{prompts}")
    TF.to_pil_image(out[0].cpu()).save(str(outpath), pnginfo=info)
    return outpath


def get_args():
    vq_parser = argparse.ArgumentParser(description="Image generation using VQGAN+CLIP")

    # Add the arguments
    vq_parser.add_argument(
        "-p", "--prompts", type=str, help="Text prompts", default=None, dest="prompts"
    )
    vq_parser.add_argument(
        "-ip",
        "--image_prompts",
        type=str,
        help="Image prompts / target image",
        default=[],
        dest="image_prompts",
    )
    vq_parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="Number of iterations",
        default=500,
        dest="max_iterations",
    )
    vq_parser.add_argument(
        "-se",
        "--save_every",
        type=int,
        help="Save image iterations",
        default=50,
        dest="display_freq",
    )
    vq_parser.add_argument(
        "-s",
        "--size",
        nargs=2,
        type=int,
        help="Image size (width height) (default: %(default)s)",
        dest="size",
    )
    vq_parser.add_argument(
        "-ii",
        "--init_image",
        type=str,
        help="Initial image",
        default=None,
        dest="init_image",
    )
    vq_parser.add_argument(
        "-in",
        "--init_noise",
        type=str,
        help="Initial noise image (pixels or gradient)",
        default=None,
        dest="init_noise",
    )
    vq_parser.add_argument(
        "-iw",
        "--init_weight",
        type=float,
        help="Initial weight",
        default=0.0,
        dest="init_weight",
    )
    vq_parser.add_argument(
        "-m",
        "--clip_model",
        type=str,
        help="CLIP model (e.g. ViT-B/32, ViT-B/16)",
        default="ViT-B/32",
        dest="clip_model",
    )
    vq_parser.add_argument(
        "-conf",
        "--vqgan_config",
        type=str,
        help="VQGAN config",
        default=f"checkpoints/vqgan_imagenet_f16_16384.yaml",
        dest="vqgan_config",
    )
    vq_parser.add_argument(
        "-ckpt",
        "--vqgan_checkpoint",
        type=str,
        help="VQGAN checkpoint",
        default=f"checkpoints/vqgan_imagenet_f16_16384.ckpt",
        dest="vqgan_checkpoint",
    )
    vq_parser.add_argument(
        "-nps",
        "--noise_prompt_seeds",
        nargs="*",
        type=int,
        help="Noise prompt seeds",
        default=[],
        dest="noise_prompt_seeds",
    )
    vq_parser.add_argument(
        "-npw",
        "--noise_prompt_weights",
        nargs="*",
        type=float,
        help="Noise prompt weights",
        default=[],
        dest="noise_prompt_weights",
    )
    vq_parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="Learning rate",
        default=0.1,
        dest="step_size",
    )
    vq_parser.add_argument(
        "-cutm",
        "--cut_method",
        type=str,
        help="Cut method",
        choices=["original", "updated", "nrupdated", "updatedpooling", "latest"],
        default="latest",
        dest="cut_method",
    )
    vq_parser.add_argument(
        "-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest="cutn"
    )
    vq_parser.add_argument(
        "-cutp",
        "--cut_power",
        type=float,
        help="Cut power",
        default=1.0,
        dest="cut_pow",
    )
    vq_parser.add_argument(
        "-sd", "--seed", type=int, help="Seed", default=None, dest="seed"
    )
    vq_parser.add_argument(
        "-opt",
        "--optimiser",
        type=str,
        help="Optimiser",
        choices=[
            "Adam",
            "AdamW",
            "Adagrad",
            "Adamax",
            "DiffGrad",
            "AdamP",
            "RAdam",
            "RMSprop",
        ],
        default="Adam",
        dest="optimiser",
    )
    vq_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output filename",
        default="output.png",
        dest="output",
    )
    vq_parser.add_argument(
        "-vid",
        "--video",
        action="store_true",
        help="Create video frames?",
        dest="make_video",
    )
    vq_parser.add_argument(
        "-zvid",
        "--zoom_video",
        action="store_true",
        help="Create zoom video?",
        dest="make_zoom_video",
    )
    vq_parser.add_argument(
        "-zs",
        "--zoom_start",
        type=int,
        help="Zoom start iteration",
        default=0,
        dest="zoom_start",
    )
    vq_parser.add_argument(
        "-zse",
        "--zoom_save_every",
        type=int,
        help="Save zoom image iterations",
        default=10,
        dest="zoom_frequency",
    )
    vq_parser.add_argument(
        "-zsc",
        "--zoom_scale",
        type=float,
        help="Zoom scale %",
        default=0.99,
        dest="zoom_scale",
    )
    vq_parser.add_argument(
        "-zsx",
        "--zoom_shift_x",
        type=int,
        help="Zoom shift x (left/right) amount in pixels",
        default=0,
        dest="zoom_shift_x",
    )
    vq_parser.add_argument(
        "-zsy",
        "--zoom_shift_y",
        type=int,
        help="Zoom shift y (up/down) amount in pixels",
        default=0,
        dest="zoom_shift_y",
    )
    vq_parser.add_argument(
        "-cpe",
        "--change_prompt_every",
        type=int,
        help="Prompt change frequency",
        default=0,
        dest="prompt_frequency",
    )
    vq_parser.add_argument(
        "-vl",
        "--video_length",
        type=float,
        help="Video length in seconds (not interpolated)",
        default=10,
        dest="video_length",
    )
    vq_parser.add_argument(
        "-ofps",
        "--output_video_fps",
        type=float,
        help="Create an interpolated video (Nvidia GPU only) with this fps (min 10. best set to 30 or 60)",
        default=30,
        dest="output_video_fps",
    )
    vq_parser.add_argument(
        "-ifps",
        "--input_video_fps",
        type=float,
        help="When creating an interpolated video, use this as the input fps to interpolate from (>0 & <ofps)",
        default=15,
        dest="input_video_fps",
    )
    vq_parser.add_argument(
        "-d",
        "--deterministic",
        action="store_true",
        help="Enable cudnn.deterministic?",
        dest="cudnn_determinism",
    )
    vq_parser.add_argument(
        "-aug",
        "--augments",
        nargs="+",
        action="append",
        type=str,
        choices=["Ji", "Sh", "Gn", "Pe", "Ro", "Af", "Et", "Ts", "Cr", "Er", "Re"],
        help="Enabled augments (latest vut method only)",
        default=[["Af", "Pe", "Ji", "Er"]],
        dest="augments",
    )
    vq_parser.add_argument(
        "-vsd",
        "--video_style_dir",
        type=str,
        help="Directory with video frames to style",
        default=None,
        dest="video_style_dir",
    )
    vq_parser.add_argument(
        "-cd",
        "--cuda_device",
        type=str,
        help="Cuda device to use",
        default="cuda:0",
        dest="cuda_device",
    )

    # Execute the parse_args() method
    args = vq_parser.parse_args("")
    return args


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    # config.model.target == 'taming.models.vqgan.VQModel':
    model = vqgan.VQModel(**config.model.params)
    model.eval().requires_grad_(False)
    model.init_from_ckpt(checkpoint_path)
    del model.loss
    return model


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, args, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow  # not used with pooling

        # Pick your own augments & their order
        augment_list = []
        for item in args.augments[0]:
            if item == "Ji":
                augment_list.append(
                    K.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7
                    )
                )
            elif item == "Sh":
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == "Gn":
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1.0, p=0.5))
            elif item == "Pe":
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == "Ro":
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == "Af":
                augment_list.append(
                    K.RandomAffine(
                        degrees=15,
                        translate=0.1,
                        shear=5,
                        p=0.7,
                        padding_mode="zeros",
                        keepdim=True,
                    )
                )  # border, reflection, zeros
            elif item == "Et":
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == "Ts":
                augment_list.append(
                    K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7)
                )
            elif item == "Cr":
                augment_list.append(
                    K.RandomCrop(
                        size=(self.cut_size, self.cut_size),
                        pad_if_needed=True,
                        padding_mode="reflect",
                        p=0.5,
                    )
                )
            elif item == "Er":
                augment_list.append(
                    K.RandomErasing(
                        scale=(0.1, 0.4),
                        ratio=(0.3, 1 / 0.3),
                        same_on_batch=True,
                        p=0.7,
                    )
                )
            elif item == "Re":
                augment_list.append(
                    K.RandomResizedCrop(
                        size=(self.cut_size, self.cut_size),
                        scale=(0.1, 1),
                        ratio=(0.75, 1.333),
                        cropping_mode="resample",
                        p=0.5,
                    )
                )

        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        # self.noise_fac = False

        # Uncomment if you like seeing the list ;)
        # print(augment_list)

        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []

        for _ in range(self.cutn):
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input)) / 2
            cutouts.append(cutout)

        batch = self.augs(torch.cat(cutouts, dim=0))

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def get_opt(opt_name, opt_lr, z):
    if opt_name == "Adam":
        opt = optim.Adam([z], lr=opt_lr)  # LR=0.1 (Default)
    elif opt_name == "AdamW":
        opt = optim.AdamW([z], lr=opt_lr)
    elif opt_name == "Adagrad":
        opt = optim.Adagrad([z], lr=opt_lr)
    elif opt_name == "Adamax":
        opt = optim.Adamax([z], lr=opt_lr)
    elif opt_name == "DiffGrad":
        opt = DiffGrad(
            [z], lr=opt_lr, eps=1e-9, weight_decay=1e-9
        )  # NR: Playing for reasons
    elif opt_name == "AdamP":
        opt = AdamP([z], lr=opt_lr)
    elif opt_name == "RAdam":
        opt = RAdam([z], lr=opt_lr)
    elif opt_name == "RMSprop":
        opt = optim.RMSprop([z], lr=opt_lr)
    else:
        print("Unknown optimiser. Are choices broken?")
        opt = optim.Adam([z], lr=opt_lr)
    return opt


def ascend_txt(i, z, perceptor, args, model, make_cutouts, pMs):
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    out = synth(z, model)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []

    if args.init_weight:
        # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
        result.append(
            F.mse_loss(z, torch.zeros_like(z_orig))
            * ((1 / torch.tensor(i * 2 + 1)) * args.init_weight)
            / 2
        )

    for prompt in pMs:
        result.append(prompt(iii))

    if args.make_video:
        img = np.array(
            out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8)
        )[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        imageio.imwrite("steps/" + str(i) + ".png", np.array(img))

    return result


def synth(z, model):
    # gumbel is False
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(
        3, 1
    )
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


def vector_quantize(x, codebook):
    d = (
        x.pow(2).sum(dim=-1, keepdim=True)
        + codebook.pow(2).sum(dim=1)
        - 2 * x @ codebook.T
    )
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


def split_prompt(prompt):
    vals = prompt.rsplit(":", 2)
    vals = vals + ["", "1", "-inf"][len(vals) :]
    return vals[0], float(vals[1]), float(vals[2])


class Prompt(nn.Module):
    def __init__(self, embed, weight=1.0, stop=float("-inf")):
        super().__init__()
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return (
            self.weight.abs()
            * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
        )
