import math
import random

import gradio as gr
import modules.scripts as scripts
from modules import deepbooru, images, processing, shared
from modules.processing import Processed
from modules.shared import opts, state

from PIL import Image
import copy

global pixmap
global xn


class Script(scripts.Script):

    def title(self):
        return "Plasma Noise"

    def show(self, is_img2img):
        if not is_img2img:
            return scripts.AlwaysVisible
        return False

    def ui(self, is_img2img):
        with gr.Accordion('Plasma Noise', open=False):
            enabled = gr.Checkbox(label="Enabled", default=False)
            turbulence = gr.Slider(minimum=0.05, maximum=10.0, step=0.05, label='Turbulence', value=2.75, elem_id=self.elem_id("turbulence"))
            denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.9, elem_id=self.elem_id("denoising"))
            noise_mult = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Noise multiplier', value=1.0, elem_id=self.elem_id("noise_mult"))

        return [enabled, turbulence, denoising, noise_mult]

    def process(self, p, enabled, turbulence, denoising, noise_mult):
        if not enabled:
            return None
            
        global pixmap
        global xn
        xn = 0
        # image size
        p.__class__ = processing.StableDiffusionProcessingImg2Img
        p.mask = None
        p.image_mask = None
        p.latent_mask = None
        p.resize_mode = None
        p.initial_noise_multiplier = noise_mult
        p.denoising_strength = denoising
        
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

        roughness = turbulence

        def adjust(xa, ya, x, y, xb, yb):
            global pixmap
            if (pixmap[x][y] == 0):
                d = math.fabs(xa - xb) + math.fabs(ya - yb)
                v = (pixmap[xa][ya] + pixmap[xb][yb]) / 2.0 + (random.random() - 0.555) * d * roughness
                c = int(math.fabs(v + (random.random() - 0.5) * 96))
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
                image.putpixel((x, y), (r[x][y], g[x][y], b[x][y]))

        p.init_images = [image]

