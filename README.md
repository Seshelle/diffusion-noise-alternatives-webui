# diffusion-noise-alternatives-webui
Creates alternative starting noise for stable diffusion txt2img.

# Usage
Only enable one noise type at a time.

Shared options between noise types:
* Graininess: Positive values add additional white noise on top of the noise after generation. Adds a grainy canvas effect to the final image.
![Graininess Comparison](https://github.com/Seshelle/diffusion-noise-alternatives-webui/blob/main/images/GrainCompare.png)
* Denoising: Like img2img denoising. A value of 0 will leave the init noise unchanged, while a value of 1 will fully denoise the init image.
* Noise multiplier: How much default latent noise to add to the image. Value of 0 adds none, while a value of 1 will apply noise at full strength.
* Level controls: Determines the maximum and minimun values of color channels in the noise. Leave at -1 for no effect. Can be used to darken, lighten, and color grade an image.
![Value Comparison](https://github.com/Seshelle/diffusion-noise-alternatives-webui/blob/main/images/ValueCompare.png)
![Color Comparison](https://github.com/Seshelle/diffusion-noise-alternatives-webui/blob/main/images/ColorGrade.png)

Plasma noise:
* Turbulence: Size/frequency of the noise. Higher values mean more high-frequency noise.

FBM noise:
* Octaves: The number of different frequencies/detail levels in the noise.
* Smoothing: Determines the pixel size of the smallest octave of noise. Larger values result in blobby and less grainy noise.
* Octave division: The frequency difference between octaves. This is almost always 2 in most FBM noise generators, but you can mess with it if you want.

# Known Issues

* Does not work with the UniPC sampler. This uses the img2img pipeline, which does not work with UniPc. The reason why you can use UniPC with img2img is because A1111 secretly changes your sampler to DDIM!!
