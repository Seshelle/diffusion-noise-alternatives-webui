import math
import random

import gradio as gr
import modules.scripts as scripts
from modules import devices, deepbooru, images, processing, shared
from modules.processing import Processed
from modules.shared import opts, state

import PIL.Image
from PIL import Image
import copy

class Script(scripts.Script):

    def __init__(self):
        self.scalingW = 0
        self.scalingH = 0
        self.hr_denoise = 0
        self.hr_steps = 0
        self.scaler = ""

    def title(self):
        return "FBM Noise"

    def show(self, is_img2img):
        if not is_img2img:
            return scripts.AlwaysVisible
        return False

    def ui(self, is_img2img):
        with gr.Accordion('FBM Noise', open=False):
            enabled = gr.Checkbox(label="Enabled", default=False)
            octaves = gr.Slider(minimum=1, maximum=32, step=1, label='Octaves', value=6, elem_id=self.elem_id("octaves"))
            smoothing = gr.Slider(minimum=1, maximum=100, step=1, label='Smoothing', value=1, elem_id=self.elem_id("smoothing"))
            grain = gr.Slider(minimum=0, maximum=256, step=1, label='Graininess', value=0, elem_id=self.elem_id("FBM_grain"))
            octave_division = gr.Slider(minimum=1.0, maximum=10.0, step=0.01, label='Octave Division', value=2, elem_id=self.elem_id("octave_division"))
            denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.9, elem_id=self.elem_id("FBM_denoising"))
            noise_mult = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Noise multiplier', value=1.0, elem_id=self.elem_id("FBM_noise_mult"))
            with gr.Accordion('Level controls', open=False):
                val_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Value Min",
                                    elem_id=self.elem_id("FBM_val_min"))
                val_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Value Max",
                                    elem_id=self.elem_id("FBM_val_max"))
                red_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Red Min",
                                    elem_id=self.elem_id("FBM_red_min"))
                red_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Red Max",
                                    elem_id=self.elem_id("FBM_red_max"))
                grn_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Green Min",
                                    elem_id=self.elem_id("FBM_grn_min"))
                grn_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Green Max",
                                    elem_id=self.elem_id("FBM_grn_max"))
                blu_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Blue Min",
                                    elem_id=self.elem_id("FBM_blu_min"))
                blu_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Blue Max",
                                    elem_id=self.elem_id("FBM_blu_max"))

        return [enabled, octaves, smoothing, grain, octave_division, denoising, noise_mult, val_min, val_max, red_min, red_max, grn_min, grn_max,
                blu_min, blu_max]

    def process(self, p, enabled, octaves, smoothing, grain, octave_division, denoising, noise_mult, val_min, val_max, red_min, red_max, grn_min,
                grn_max, blu_min, blu_max):
        if not enabled or "alt_hires" in p.extra_generation_params:
            return None

        if p.enable_hr:
            self.hr_denoise = p.denoising_strength
            self.hr_steps = p.hr_second_pass_steps
            if self.hr_steps == 0:
                self.hr_steps = p.steps
            if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                self.scalingW = p.hr_scale
                self.scalingH = p.hr_scale
            else:
                self.scalingW = p.hr_resize_x
                self.scalingH = p.hr_resize_y
            self.scaler = p.hr_upscaler
        else:
            self.scalingW = 0

        p.__class__ = processing.StableDiffusionProcessingImg2Img
        p.mask = None
        p.image_mask = None
        p.latent_mask = None
        p.resize_mode = None
        p.inpaint_full_res = None
        p.extra_generation_params["Alt noise type"] = "FBM"
        p.extra_generation_params["Octaves"] = octaves
        p.extra_generation_params["Smoothing"] = smoothing
        p.extra_generation_params["Alt denoising strength"] = denoising
        p.extra_generation_params["Alt noise multiplier"] = noise_mult
        p.extra_generation_params["Value Min"] = val_min
        p.extra_generation_params["Value Max"] = val_max
        p.extra_generation_params["Red Min"] = red_min
        p.extra_generation_params["Red Max"] = red_max
        p.extra_generation_params["Green Min"] = grn_min
        p.extra_generation_params["Green Max"] = grn_max
        p.extra_generation_params["Blue Min"] = blu_min
        p.extra_generation_params["Blue Max"] = blu_max

        p.initial_noise_multiplier = noise_mult
        p.denoising_strength = float(denoising)

        random.seed(p.all_seeds[0])

        width = p.width
        height = p.height
        square = max(width, height)
        max_octaves = 1
        octave_pixel_size = 1
        while True:
            octave_pixel_size *= octave_division
            if octave_pixel_size < square / octave_division:
                max_octaves += 1
                if max_octaves >= octaves:
                    break
            else:
                break
        octaves = min(octaves, max_octaves)

        mr = max(val_min, red_min, 0)
        mg = max(val_min, grn_min, 0)
        mb = max(val_min, blu_min, 0)
        hv = 255
        if val_max >= 0:
            hv = val_max
        hr = 255
        if red_max >= 0:
            hr = val_max
        hg = 255
        if grn_max >= 0:
            hg = val_max
        hb = 255
        if blu_max >= 0:
            hb = val_max
        red_range = min(hv, hr) - mr
        green_range = min(hv, hg) - mg
        blue_range = min(hv, hb) - mb

        if grain > 0:
            grain_image_r = [[0 for i in range(height)] for j in range(width)]
            grain_image_g = [[0 for i in range(height)] for j in range(width)]
            grain_image_b = [[0 for i in range(height)] for j in range(width)]
            for y in range(height):
                for x in range(width):
                    grain_image_r[x][y] = int((random.random() - 0.5) * grain)
                    grain_image_g[x][y] = int((random.random() - 0.5) * grain)
                    grain_image_b[x][y] = int((random.random() - 0.5) * grain)

        final_image = Image.new("RGB", (width, height))

        for o in range(octaves):
            a = smoothing * pow(octave_division, octaves - o - 1)

            s = int(square / a)
            if s > square:
                break

            r = [[0 for i in range(s)] for j in range(s)]
            g = [[0 for i in range(s)] for j in range(s)]
            b = [[0 for i in range(s)] for j in range(s)]

            octave_image = Image.new("RGB", (s, s))
            for y in range(s):
                for x in range(s):
                    r[x][y] = int(random.random() * red_range + mr)
                    g[x][y] = int(random.random() * green_range + mg)
                    b[x][y] = int(random.random() * blue_range + mb)

            for y in range(s):
                for x in range(s):
                    octave_image.putpixel((x, y), (r[x][y], g[x][y], b[x][y]))

            octave_image = octave_image.resize((square, square), PIL.Image.BILINEAR)

            for y in range(height):
                for x in range(width):
                    old_pix = final_image.getpixel((x, y))
                    new_pix = octave_image.getpixel((x, y))
                    amplitude = 1 / pow(2, o + 1)
                    new_pix = (int(new_pix[0] * amplitude), int(new_pix[1] * amplitude), int(new_pix[2] * amplitude))
                    if grain > 0 and o == octaves - 1:
                        final_pix = (max(min(old_pix[0] + new_pix[0] + grain_image_r[x][y], hr), 0),
                        max(min(old_pix[1] + new_pix[1] + grain_image_g[x][y], hr), 0),
                        max(min(old_pix[2] + new_pix[2] + grain_image_b[x][y], hr), 0))
                    else:
                        final_pix = (old_pix[0] + new_pix[0], old_pix[1] + new_pix[1], old_pix[2] + new_pix[2])
                    final_image.putpixel((x, y), final_pix)

        p.init_images = [final_image]

    def postprocess(self, p, processed, enabled, octaves, smoothing, grain, octave_division, denoising, noise_mult, val_min, val_max, red_min, red_max, grn_min,
                grn_max, blu_min, blu_max):
        if not enabled or self.scalingW == 0 or "alt_hires" in p.extra_generation_params or not p.enable_hr:
            return None
        devices.torch_gc()

        for i in range(len(processed.images)):
            p.init_images[i] = processed.images[i]

        p.extra_generation_params["alt_hires"] = self.scalingW
        p.width = int(p.width * self.scalingW)
        p.height = int(p.height * self.scalingH)
        p.denoising_strength = self.hr_denoise
        p.steps = self.hr_steps
        p.resize_mode = 3 if 'Latent' in self.scaler else 0
        new_p = processing.process_images(p)
        processed.images = new_p.images
