# File: ai_image_generator.py
import torch
from diffusers import StableDiffusionPipeline
import json
from typing import List, Dict

class RelationshipImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        )
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
        
        # Tạo prompt từ relationship
        base_prompt = self.relationship_templates.get(relation, f"{subject} {relation} {obj}")
        
        # Tạo các biến thể
        variations = self.create_variations(base_prompt, num_variations)
        
        # Generate images
        generated_images = []
        for prompt in variations:
            image = self.pipe(prompt, num_inference_steps=50).images[0]
            generated_images.append({
                'image': image,
                'prompt': prompt,
                'original_relationship': relationship
            })
        
        return generated_images
    
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