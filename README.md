# diffusion-noise-alternatives-webui
Creates alternative starting noise for stable diffusion txt2img.

# Usage
Only enable one noise type at a time.

Shared options between noise types:
* Graininess: Positive values add additional white noise on top of the noise after generation. Adds a grainy canvas effect to the final image.
![Graininess Comparison](https://github.com/Seshelle/diffusion-noise-alternatives-webui/blob/main/images/GrainCompare.png)
* Denoising: Like img2img denoising. A value of 0 will leave the init noise unchanged, while a value of 1 will fully denoise the init image.
* Noise multiplier: How much default latent noise to add to the image. Value of 0 adds none, while a value of 1 will apply noise at full strength.
* Level controls: Can be used to darken, lighten, and color grade an image to create images not normally possible with stable diffusion. Options determine the maximum and minimun values of the RGB values in the noise. Brightness controls the value of all color channels, unless overriden by a specific RGB option. Leave options at -1 for default 0-256 range.
![Value Comparison](https://github.com/Seshelle/diffusion-noise-alternatives-webui/blob/main/images/ValueCompare.png)
![Color Comparison](https://github.com/Seshelle/diffusion-noise-alternatives-webui/blob/main/images/ColorGrade.png)
* Contrast: Values above 1 make the brighter parts of the noise brighter and the darker parts darker. Values below 1 make the image more gray.
* Greyscale: Makes the init noise black and white by using only the red channel of the image for all color channels.
![Contrast Comparison](https://github.com/Seshelle/diffusion-noise-alternatives-webui/blob/main/images/contrast.png)

Plasma noise:
* Turbulence: Size/frequency of the noise. Higher values mean more high-frequency noise.

FBM noise:
* Octaves: The number of different frequencies/detail levels in the noise.
* Smoothing: Determines the pixel size of the smallest octave of noise. Larger values result in blobby and less grainy noise.
* Octave division: The frequency difference between octaves. This is almost always 2 in most FBM noise generators, but you can mess with it if you want.

# Known Issues

* Does not work with the UniPC sampler. This uses the img2img pipeline, which does not work with UniPc. The reason why you can use UniPC with img2img is because A1111 secretly changes your sampler to DDIM!!
