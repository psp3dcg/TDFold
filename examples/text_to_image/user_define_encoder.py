from transformers import AutoModel, AutoTokenizer
from diffusers import StableDiffusionPipeline
#import torch

text_encoder = AutoModel.from_pretrained("StanfordAIMI/RadBERT")#, load_in_8bit=True)
text_tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/RadBERT")#, load_in_8bit=True)

text_prompts = ["A Chest X-ray of a patient with pneumonia."]

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
													text_encoder=text_encoder, safety_checker=None,
													#torch_dtype=torch.float16, variant='fp16'
													).to("cuda")
pipeline.tokenizer = text_tokenizer
pipeline.tokenizer.model_max_length = 512

image = pipeline(prompt=text_prompts,
			     height=512,
			     width=512,
			     num_inference_steps=25,
			     num_images_per_prompt=1,
			     ).images[0]
image