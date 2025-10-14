# File: ai_image_generator.py
import torch
import json
from typing import List, Dict

# Try to import diffusers, fallback to mock if not available
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("WARNING: diffusers not available, using mock generator")
    DIFFUSERS_AVAILABLE = False

class RelationshipImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        print("Initializing AI Image Generator...")
        
        if not DIFFUSERS_AVAILABLE:
            print("WARNING: diffusers not available, using mock generator")
            self.pipe = None
        else:
            try:
                # Check if CUDA is available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")
                
                print("Loading Stable Diffusion model... This may take a while...")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=True
                )
                self.pipe = self.pipe.to(device)
                print("SUCCESS: Stable Diffusion model loaded successfully!")
            except Exception as e:
                print(f"ERROR: Failed to load Stable Diffusion model: {e}")
                print("Falling back to mock generator")
                self.pipe = None
        
        self.relationship_templates = self.load_relationship_templates()
    
    def load_relationship_templates(self):
        return {
            'holding': "{subject} holding {object} in hands",
            'sitting_on': "{subject} sitting on {object}",
            'near': "{subject} standing near {object}",
            'wearing': "{subject} wearing {object}",
            'looking_at': "{subject} looking at {object}",
            'behind': "{subject} behind {object}",
            'in_front_of': "{subject} in front of {object}"
        }
    
    def generate_from_relationship(self, relationship: Dict, num_variations: int = 5):
        subject = relationship['subject']
        relation = relationship['relation']
        obj = relationship['object']
        
        print(f"Generating images for: {subject} {relation} {obj}")
        
        # Tạo prompt từ relationship
        base_prompt = self.relationship_templates.get(relation, f"{subject} {relation} {obj}")
        
        # Tạo các biến thể
        variations = self.create_variations(base_prompt, num_variations)
        
        # Generate images
        generated_images = []
        
        if self.pipe is None:
            print("Using mock image generation (Stable Diffusion not available)")
            # Mock generation - tạo fake data
            for i, prompt in enumerate(variations):
                mock_image = self.create_mock_image()
                generated_images.append({
                    'image': mock_image,
                    'prompt': prompt,
                    'original_relationship': relationship,
                    'is_mock': True
                })
        else:
            print(f"Generating {len(variations)} images with Stable Diffusion...")
            for i, prompt in enumerate(variations):
                print(f"  Generating image {i+1}/{len(variations)}: {prompt[:50]}...")
                try:
                    image = self.pipe(prompt, num_inference_steps=20).images[0]  # Giảm steps để nhanh hơn
                    generated_images.append({
                        'image': image,
                        'prompt': prompt,
                        'original_relationship': relationship,
                        'is_mock': False
                    })
                except Exception as e:
                    print(f"ERROR: Error generating image {i+1}: {e}")
                    # Fallback to mock
                    mock_image = self.create_mock_image()
                    generated_images.append({
                        'image': mock_image,
                        'prompt': prompt,
                        'original_relationship': relationship,
                        'is_mock': True
                    })
        
        print(f"SUCCESS: Generated {len(generated_images)} images")
        return generated_images
    
    def create_mock_image(self):
        """Create a mock image for testing"""
        from PIL import Image
        import numpy as np
        
        # Tạo ảnh mock đơn giản
        mock_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mock_image = Image.fromarray(mock_array)
        return mock_image
    
    def create_variations(self, base_prompt: str, num_variations: int):
        variations = []
        
        # Lighting variations
        lighting_styles = [
            "bright daylight", "soft lighting", "golden hour", 
            "blue hour", "studio lighting", "natural lighting"
        ]
        
        # Style variations  
        styles = [
            "photorealistic", "high quality", "detailed", 
            "professional photography", "sharp focus"
        ]
        
        # Context variations
        contexts = [
            "indoor setting", "outdoor setting", "urban environment",
            "natural environment", "modern setting"
        ]
        
        for i in range(num_variations):
            lighting = lighting_styles[i % len(lighting_styles)]
            style = styles[i % len(styles)]
            context = contexts[i % len(contexts)]
            
            variation = f"{base_prompt}, {style}, {lighting}, {context}"
            variations.append(variation)
        
        return variations