from diffusers import StableDiffusionPipeline
import torch
import requests
from mobilegan import MobileGAN
from fastgan import FastGAN
from ldm.models.diffusion.ddpm import LatentDiffusion
from qnnx import QNNXModel
import tensorflow as tf
from PIL import Image
import numpy as np

##########################################################################################
# Load pre-trained model
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# pipeline.to("cuda")  # Use GPU for faster processing

# Generate an image
prompt = "A futuristic cityscape at sunset"
generated_image = pipeline(prompt).images[0]
generated_image.save("output.png")

##########################################################################################
# Load a different pre-trained model
lite_pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-lite")

# Generate an image using the new model
lite_prompt = "A serene mountain landscape with a clear blue lake"
lite_generated_image = lite_pipeline(lite_prompt).images[0]
lite_generated_image.save("lite_output.png")

##########################################################################################
# Generate an image using DeepAI's text-to-image API
deepai_api_url = "https://api.deepai.org/api/text2img"
deepai_api_key = "your_deepai_api_key"  # Replace with your actual API key

deepai_prompt = "A magical forest with glowing mushrooms"
response = requests.post(
    deepai_api_url,
    data={"text": deepai_prompt},
    headers={"api-key": deepai_api_key}
)

if response.status_code == 200:
    result = response.json()
    image_url = result.get("output_url")
    if image_url:
        print(f"Generated image URL: {image_url}")
    else:
        print("Failed to retrieve the image URL.")
else:
    print(f"Error: {response.status_code}, {response.text}")

##########################################################################################
# Generate an image using MobileGAN
##########################################################################################
# Initialize MobileGAN model
mobilegan_model = MobileGAN(pretrained=True)

# Define the prompt for MobileGAN
mobilegan_prompt = "A vibrant underwater coral reef scene"

# Generate the image
mobilegan_generated_image = mobilegan_model.generate_image(mobilegan_prompt)

# Save the generated image
mobilegan_generated_image.save("mobilegan_output.png")

##########################################################################################
# Generate an image using FastGAN
##########################################################################################
# Initialize FastGAN model
fastgan_model = FastGAN(pretrained=True)

# Define the prompt for FastGAN
fastgan_prompt = "A cozy cabin in a snowy forest"

# Generate the image
fastgan_generated_image = fastgan_model.generate_image(fastgan_prompt)

# Save the generated image
fastgan_generated_image.save("fastgan_output.png")

##########################################################################################
# Generate an image using Latent Diffusion Models (LDM)
##########################################################################################
# Initialize LDM model
ldm_model = LatentDiffusion.from_pretrained("CompVis/ldm-text2im-large-256")

# Define the prompt for LDM
ldm_prompt = "A futuristic robot in a neon-lit city"

# Generate the image
ldm_generated_image = ldm_model.generate_image(ldm_prompt)

# Save the generated image
ldm_generated_image.save("ldm_output.png")

##########################################################################################
# Generate an image using QNNX-Converted Model
##########################################################################################
# Initialize QNNX model
qnnx_model = QNNXModel.load_pretrained("path/to/qnnx-converted-model")

# Define the prompt for QNNX model
qnnx_prompt = "A majestic eagle soaring over a canyon at sunrise"

# Generate the image
qnnx_generated_image = qnnx_model.generate_image(qnnx_prompt)

# Save the generated image
qnnx_generated_image.save("qnnx_output.png")

##########################################################################################
# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="path/to/tflite_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the prompt for TensorFlow Lite model
tflite_prompt = "A peaceful beach with palm trees at sunset"

# Preprocess the prompt (convert to input tensor format)
# Assuming the model expects a text embedding or similar input
# Replace this with the actual preprocessing required for your model
input_data = np.array([tflite_prompt], dtype=np.float32)  # Example placeholder

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Postprocess the output to generate an image
# Assuming the output is an image tensor
image_array = np.squeeze(output_data)  # Remove batch dimension if present
image = Image.fromarray((image_array * 255).astype(np.uint8))  # Convert to image

# Save the generated image
image.save("tflite_output.png")