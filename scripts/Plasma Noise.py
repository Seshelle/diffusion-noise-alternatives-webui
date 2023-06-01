import math
import random

import gradio as gr
import modules.scripts as scripts
from modules import devices, deepbooru, images, processing, shared
from modules.processing import Processed
from modules.shared import opts, state

from PIL import Image
import copy

global pixmap
global xn


class Script(scripts.Script):

    def __init__(self):
        self.scalingW = 0
        self.scalingH = 0
        self.hr_denoise = 0
        self.hr_steps = 0
        self.scaler = ""

    def title(self):
        return "Plasma Noise"

    def show(self, is_img2img):
        if not is_img2img:
            return scripts.AlwaysVisible
        return False

    def ui(self, is_img2img):
        with gr.Accordion('Plasma Noise', open=False):
            enabled = gr.Checkbox(label="Enabled", default=False)
            turbulence = gr.Slider(minimum=0.05, maximum=10.0, step=0.05, label='Turbulence', value=2.75,
                                   elem_id=self.elem_id("turbulence"))
            grain = gr.Slider(minimum=0, maximum=256, step=1, label='Graininess', value=96,
                                   elem_id=self.elem_id("plasma_grain"))
            denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.9,
                                  elem_id=self.elem_id("denoising"))
            noise_mult = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Noise multiplier', value=1.0,
                                   elem_id=self.elem_id("noise_mult"))
            with gr.Accordion('Level controls', open=False):
                val_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Value Min",
                                    elem_id=self.elem_id("plasma_val_min"))
                val_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Value Max",
                                    elem_id=self.elem_id("plasma_val_max"))
                red_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Red Min",
                                    elem_id=self.elem_id("plasma_red_min"))
                red_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Red Max",
                                    elem_id=self.elem_id("plasma_red_max"))
                grn_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Green Min",
                                    elem_id=self.elem_id("plasma_grn_min"))
                grn_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Green Max",
                                    elem_id=self.elem_id("plasma_grn_max"))
                blu_min = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Blue Min",
                                    elem_id=self.elem_id("plasma_blu_min"))
                blu_max = gr.Slider(minimum=-1, maximum=255, step=1, value=-1, label="Blue Max",
                                    elem_id=self.elem_id("plasma_blu_max"))

        return [enabled, turbulence, grain, denoising, noise_mult, val_min, val_max, red_min, red_max, grn_min, grn_max,
                blu_min, blu_max]

    def process(self, p, enabled, turbulence, grain, denoising, noise_mult, val_min, val_max, red_min, red_max, grn_min,
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

        global pixmap
        global xn
        xn = 0
        # image size
        p.__class__ = processing.StableDiffusionProcessingImg2Img
        p.mask = None
        p.image_mask = None
        p.latent_mask = None
        p.resize_mode = None
        p.inpaint_full_res = None
        p.extra_generation_params["Alt noise type"] = "Plasma"
        p.extra_generation_params["Turbulence"] = turbulence
        p.extra_generation_params["Grain"] = grain
        p.extra_generation_params["Alt denoising strength"] = denoising
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

        w = p.width
        h = p.height
        processing.fix_seed(p)
        random.seed(p.seed)
        aw = copy.deepcopy(w)
        ah = copy.deepcopy(h)
        image = Image.new("RGB", (aw, ah))
        if w >= h:
            h = w
        else:
            w = h

        # Clamp per channel and globally
        clamp_v_min = val_min
        clamp_v_max = val_max
        clamp_r_min = red_min
        clamp_r_max = red_max
        clamp_g_min = grn_min
        clamp_g_max = grn_max
        clamp_b_min = blu_min
        clamp_b_max = blu_max

        # Handle value clamps
        lv = 0
        mv = 0
        if clamp_v_min == -1:
            lv = 0
        else:
            lv = clamp_v_min

        if clamp_v_max == -1:
            mv = 255
        else:
            mv = clamp_v_max

        lr = 0
        mr = 0
        if clamp_r_min == -1:
            lr = lv
        else:
            lr = clamp_r_min

        if clamp_r_max == -1:
            mr = mv
        else:
            mr = clamp_r_max

        lg = 0
        mg = 0
        if clamp_g_min == -1:
            lg = lv
        else:
            lg = clamp_g_min

        if clamp_g_max == -1:
            mg = mv
        else:
            mg = clamp_g_max

        lb = 0
        mb = 0
        if clamp_b_min == -1:
            lb = lv
        else:
            lb = clamp_b_min

        if clamp_b_max == -1:
            mb = mv
        else:
            mb = clamp_b_max

        roughness = turbulence

        def remap(v, low2, high2):
            # low2 + (value - low1) * (high2 - low2) / (high1 - low1)
            return int(low2 + v * (high2 - low2) / (255))

        def adjust(xa, ya, x, y, xb, yb):
            global pixmap
            if (pixmap[x][y] == 0):
                d = math.fabs(xa - xb) + math.fabs(ya - yb)
                v = (pixmap[xa][ya] + pixmap[xb][yb]) / 2.0 + (random.random() - 0.555) * d * roughness
                c = int(math.fabs(v + (random.random() - 0.5) * grain))
                if c < 0:
                    c = 0
                elif c > 255:
                    c = 255
                pixmap[x][y] = c

        def subdivide(x1, y1, x2, y2):
            global pixmap
            if (not ((x2 - x1 < 2.0) and (y2 - y1 < 2.0))):
                x = int((x1 + x2) / 2.0)
                y = int((y1 + y2) / 2.0)
                adjust(x1, y1, x, y1, x2, y1)
                adjust(x2, y1, x2, y, x2, y2)
                adjust(x1, y2, x, y2, x2, y2)
                adjust(x1, y1, x1, y, x1, y2)
                if (pixmap[x][y] == 0):
                    v = int((pixmap[x1][y1] + pixmap[x2][y1] + pixmap[x2][y2] + pixmap[x1][y2]) / 4.0)
                    pixmap[x][y] = v

                subdivide(x1, y1, x, y)
                subdivide(x, y1, x2, y)
                subdivide(x, y, x2, y2)
                subdivide(x1, y, x, y2)

        pixmap = [[0 for i in range(h)] for j in range(w)]
        pixmap[0][0] = int(random.random() * 255)
        pixmap[w - 1][0] = int(random.random() * 255)
        pixmap[w - 1][h - 1] = int(random.random() * 255)
        pixmap[0][h - 1] = int(random.random() * 255)
        subdivide(0, 0, w - 1, h - 1)
        r = copy.deepcopy(pixmap)

        pixmap = [[0 for i in range(h)] for j in range(w)]
        pixmap[0][0] = int(random.random() * 255)
        pixmap[w - 1][0] = int(random.random() * 255)
        pixmap[w - 1][h - 1] = int(random.random() * 255)
        pixmap[0][h - 1] = int(random.random() * 255)
        subdivide(0, 0, w - 1, h - 1)
        g = copy.deepcopy(pixmap)

        pixmap = [[0 for i in range(h)] for j in range(w)]
        pixmap[0][0] = int(random.random() * 255)
        pixmap[w - 1][0] = int(random.random() * 255)
        pixmap[w - 1][h - 1] = int(random.random() * 255)
        pixmap[0][h - 1] = int(random.random() * 255)
        subdivide(0, 0, w - 1, h - 1)
        b = copy.deepcopy(pixmap)

        for y in range(ah):
            for x in range(aw):
                image.putpixel((x, y), (remap(r[x][y], lr, mr), remap(g[x][y], lg, mg), remap(b[x][y], lb, mb)))

        p.init_images = [image]

    def postprocess(self, p, processed, enabled, turbulence, grain, denoising, noise_mult, val_min, val_max, red_min, red_max, grn_min,
                grn_max, blu_min, blu_max):
        """if not enabled or self.scalingW == 0 or "alt_hires" in p.extra_generation_params:
            return None
        for i in range(len(processed.images)):
            p.init_images[i] = (images.resize_image(0, processed.images[i], p.width * self.scalingW, p.height * self.scalingH, self.scaler))

        p.extra_generation_params["alt_hires"] = self.scalingW
        p.width = int(p.width * self.scalingW)
        p.height = int(p.height * self.scalingH)
        p.denoising_strength = float(self.hr_denoise)
        p.steps = int(self.hr_steps)
        devices.torch_gc()
        new_p = processing.process_images(p)
        processed.images = new_p.images"""
        if not enabled or self.scalingW == 0 or "alt_hires" in p.extra_generation_params or not p.enable_hr:
            return None
        devices.torch_gc()
        latent_scale_mode = shared.latent_upscale_modes.get(self.scaler, None) if self.scaler is not None else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "nearest")
        for i in range(len(processed.images)):
            if 'Latent' in self.scaler:
                p.latent_scale_mode = latent_scale_mode
                p.init_images[i] = processed.images[i]
            else:
                p.init_images[i] = (images.resize_image(0, processed.images[i], int(p.width * self.scalingW),
                                                        int(p.height * self.scalingH), self.scaler))

        p.extra_generation_params["alt_hires"] = self.scalingW
        p.width = int(p.width * self.scalingW)
        p.height = int(p.height * self.scalingH)
        p.denoising_strength = self.hr_denoise
        p.steps = self.hr_steps
        p.resize_mode = 1 if 'Latent' in self.scaler else self.scaler
        new_p = processing.process_images(p)
        processed.images = new_p.images
