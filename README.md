# diffustion-noise-alternatives-webui
Creates alternative starting noise for stable diffusion txt2img, in this case plasma noise.

# Usage

* Enabled: Will replace init noise with alternative noise when on. Script does nothing when this is off.
* Turbulence: Size/frequency of the noise. Higher values mean more high-frequency noise.
* Denoising: Like img2img denoising. A value of 0 will leave the init noise unchanged, while a value of 1 will fully denoise the init image.
* Noise multiplier: How much default latent noise to add to the image. Value of 0 adds none, while a value of 1 will apply noise at full strength.
