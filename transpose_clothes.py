import os
import glob
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import openai
import requests
from dotenv import load_dotenv
import time
import base64

class ClothesTransposer:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.screenshot_dir = Path('screenshots')
        self.reference_dir = Path('references')
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)

    def get_latest_image(self, directory):
        """Get the most recent image from a directory."""
        files = glob.glob(str(directory / '*'))
        if not files:
            raise FileNotFoundError(f"No images found in {directory}")
        return max(files, key=os.path.getctime)

    def validate_images(self, screenshot_path, reference_path):
        """Validate input images for quality and compatibility."""
        try:
            # Check if images exist
            if not os.path.exists(screenshot_path) or not os.path.exists(reference_path):
                raise FileNotFoundError("One or both images not found")

            # Read images
            screenshot = cv2.imread(screenshot_path)
            reference = cv2.imread(reference_path)

            if screenshot is None or reference is None:
                raise ValueError("Failed to read one or both images")

            # Check image dimensions
            if screenshot.shape[0] < 256 or screenshot.shape[1] < 256:
                raise ValueError("Screenshot image too small")
            if reference.shape[0] < 256 or reference.shape[1] < 256:
                raise ValueError("Reference image too small")

            return True

        except Exception as e:
            print(f"Image validation error: {e}")
            return False

    def transpose_clothes(self):
        """Main function to transpose clothes from screenshot to reference image."""
        try:
            # Get latest images
            screenshot_path = self.get_latest_image(self.screenshot_dir)
            reference_path = self.get_latest_image(self.reference_dir)

            # Validate images
            if not self.validate_images(screenshot_path, reference_path):
                return None

            # Start timing
            start_time = time.time()
            print("Starting image generation...")

            # Generate the image using OpenAI's API
            response = self.client.images.generate(
                model="gpt-image-1",
                prompt=f"""Create a photorealistic full-body image that perfectly combines:
1. The exact clothes, including all details, patterns, and textures from the outfit in this screenshot: {screenshot_path}
2. The exact body, face, hair, and pose from this reference image: {reference_path}

Critical Requirements:
- Maintain the EXACT same body proportions, height, and build as the reference image
- Keep the EXACT same face, facial features, and hair from the reference image
- Preserve the EXACT same pose and body stance as the reference image
- Transfer the clothes with perfect accuracy, including all details, patterns, and textures
- Ensure the clothes fit naturally on the body, maintaining proper proportions
- Show the complete full body from head to toe in the frame
- Use a clean white background
- Maintain photorealistic quality
- Ensure the clothes look exactly like they do in the screenshot, with no modifications to style or design
- Keep all body measurements and proportions identical to the reference image
- Preserve all facial features, expressions, and hair details exactly as in the reference
- Make sure the clothes appear to be worn naturally, with proper fit and drape

The final image should look like the reference person wearing the exact clothes from the screenshot, with perfect body matching and natural clothing fit.""",
                n=1,
                size="1024x1024"
            )

            # Calculate generation time
            generation_time = time.time() - start_time
            print(f"Image generation completed in {generation_time:.2f} seconds")

            # Save the generated image
            output_path = self.output_dir / f"transposed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            # Download and save the image
            if not hasattr(response, 'data') or not response.data:
                print("Response data:", response)  # Debug print
                raise ValueError("Invalid response format from API")
                
            image_data = response.data[0]
            if not hasattr(image_data, 'b64_json'):
                raise ValueError("No image data received from the API")
                
            # Decode base64 image data
            image_bytes = base64.b64decode(image_data.b64_json)
            
            # Save the image
            with open(output_path, 'wb') as f:
                f.write(image_bytes)

            return str(output_path)

        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

if __name__ == "__main__":
    transposer = ClothesTransposer()
    output_path = transposer.transpose_clothes()
    if output_path:
        print(f"Successfully generated image at: {output_path}")
    else:
        print("Failed to generate image") 