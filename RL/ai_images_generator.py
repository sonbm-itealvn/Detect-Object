# File: ai_image_generator.py
import torch
import json
import random
from typing import List, Dict, Optional

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
        """
        Template mapping từ relation type sang câu mô tả tự nhiên.
        Mỗi template được thiết kế để tạo prompt rõ ràng cho Stable Diffusion.
        """
        return {
            # Spatial relationships
            'holding': "{subject} holding {object} in hands",
            'sitting_on': "{subject} sitting on {object}",
            'near': "{subject} standing near {object}",
            'behind': "{subject} behind {object}",
            'in_front_of': "{subject} in front of {object}",
            'above': "{subject} above {object}",
            'below': "{subject} below {object}",
            'next_to': "{subject} next to {object}",
            'beside': "{subject} beside {object}",
            
            # Action relationships
            'wearing': "{subject} wearing {object}",
            'looking_at': "{subject} looking at {object}",
            'carrying': "{subject} carrying {object}",
            'riding': "{subject} riding {object}",
            'pushing': "{subject} pushing {object}",
            'pulling': "{subject} pulling {object}",
            'touching': "{subject} touching {object}",
            'using': "{subject} using {object}",
            
            # State relationships
            'on': "{subject} on {object}",
            'in': "{subject} in {object}",
            'under': "{subject} under {object}",
            'over': "{subject} over {object}",
            'inside': "{subject} inside {object}",
            'outside': "{subject} outside {object}",
            
            # Default fallback
            'default': "{subject} {relation} {object}"
        }
    
    def generate_from_relationship(self, relationship: Dict, num_variations: int = 5, seed: Optional[int] = None):
        """
        Generate images từ relationship triplet (Subject, Predicate, Object).
        
        Quy trình:
        1. Lấy triplet: (subject, relation, object)
        2. Map relation sang template: "person holding phone" 
        3. Format template với subject và object: "person holding phone in hands"
        4. Tạo variations với context, lighting, background: 
           "person holding phone in hands, high quality, bright daylight, on the street"
        5. Generate images với Stable Diffusion
        
        Args:
            relationship: Dict với keys 'subject', 'relation', 'object'
            num_variations: Số lượng biến thể prompt (ảnh) cần sinh
            seed: Random seed để reproducibility
        
        Returns:
            List of generated image data (dict với 'image', 'prompt', 'original_relationship')
        """
        subject = relationship.get('subject', 'unknown')
        relation = relationship.get('relation', 'unknown')
        obj = relationship.get('object', 'unknown')
        
        print(f"Generating images for: {subject} {relation} {obj}")
        
        # Bước 1: Tạo base prompt từ relationship template
        # Template mapping: relation -> "{subject} {relation_verb} {object}"
        template = self.relationship_templates.get(relation, self.relationship_templates.get('default', "{subject} {relation} {object}"))
        base_prompt = template.format(subject=subject, object=obj, relation=relation)
        
        print(f"  Base prompt: {base_prompt}")
        
        # Bước 2: Tạo các biến thể với prompt engineering
        # Thêm quality, lighting, background, context để tăng đa dạng Sdiv
        variations = self.create_variations(base_prompt, num_variations, seed=seed)
        
        print(f"  Generated {len(variations)} prompt variations")
        
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
    
    def create_variations(self, base_prompt: str, num_variations: int, seed: Optional[int] = None):
        """
        Tạo các biến thể prompt đa dạng để đảm bảo tính đa dạng Sdiv.
        
        Strategy:
        1. Quality modifiers: Đảm bảo chất lượng ảnh cao
        2. Lighting variations: Thay đổi ánh sáng để đa dạng
        3. Background/Location: Thêm context về địa điểm
        4. Style modifiers: Thay đổi phong cách
        5. Random sampling: Tránh cyclic pattern, tăng đa dạng
        
        Args:
            base_prompt: Prompt cơ bản từ relationship template
            num_variations: Số lượng biến thể cần tạo
            seed: Random seed để reproducibility (optional)
        
        Returns:
            List of prompt strings
        """
        if seed is not None:
            random.seed(seed)
        
        variations = []
        
        # ============================================
        # 1. QUALITY MODIFIERS (luôn có để đảm bảo chất lượng)
        # ============================================
        quality_modifiers = [
            "high quality", "photorealistic", "detailed", 
            "professional photography", "sharp focus", "4k",
            "ultra detailed", "best quality", "masterpiece"
        ]
        
        # ============================================
        # 2. LIGHTING VARIATIONS (tăng đa dạng visual)
        # ============================================
        lighting_styles = [
            "bright daylight", "soft natural lighting", "golden hour lighting",
            "blue hour lighting", "studio lighting", "dramatic lighting",
            "sunset lighting", "morning light", "evening light",
            "overcast lighting", "sunny day", "shaded area"
        ]
        
        # ============================================
        # 3. BACKGROUND/LOCATION CONTEXT (quan trọng cho Sdiv)
        # ============================================
        background_contexts = [
            # Indoor
            "in a room", "indoors", "inside a building", "in an office",
            "in a kitchen", "in a living room", "in a bedroom",
            
            # Outdoor - Urban
            "on the street", "on a city street", "in an urban area",
            "on a sidewalk", "in a parking lot", "on a road",
            "in a city", "downtown", "in a public place",
            
            # Outdoor - Natural
            "in a park", "in nature", "outdoors", "in a forest",
            "on a field", "in a garden", "near trees",
            "on a beach", "by the water", "in a natural setting",
            
            # Specific locations
            "at a crosswalk", "at a bus stop", "in a shopping area",
            "on a bridge", "in a plaza", "at a market"
        ]
        
        # ============================================
        # 4. TIME/WEATHER CONTEXT (thêm chi tiết)
        # ============================================
        time_weather = [
            "during daytime", "during night", "in the morning",
            "in the afternoon", "in the evening", "clear weather",
            "sunny day", "cloudy day", "rainy day"
        ]
        
        # ============================================
        # 5. STYLE DESCRIPTORS (tùy chọn, không quá nhiều)
        # ============================================
        style_descriptors = [
            "realistic", "lifelike", "natural", "authentic",
            "candid", "documentary style"
        ]
        
        # ============================================
        # 6. COMPOSITION HINTS (giúp Stable Diffusion hiểu rõ hơn)
        # ============================================
        composition_hints = [
            "full body", "full shot", "wide angle", "medium shot",
            "clear view", "well composed", "centered"
        ]
        
        # Tạo variations với random sampling để đảm bảo đa dạng
        used_combinations = set()
        
        for i in range(num_variations):
            # Chọn ngẫu nhiên các components
            quality = random.choice(quality_modifiers)
            lighting = random.choice(lighting_styles)
            background = random.choice(background_contexts)
            
            # Có 70% chance thêm time/weather
            time_weather_str = ""
            if random.random() < 0.7:
                time_weather_str = f", {random.choice(time_weather)}"
            
            # Có 50% chance thêm style descriptor
            style_str = ""
            if random.random() < 0.5:
                style_str = f", {random.choice(style_descriptors)}"
            
            # Có 40% chance thêm composition hint
            composition_str = ""
            if random.random() < 0.4:
                composition_str = f", {random.choice(composition_hints)}"
            
            # Tạo prompt theo format chuẩn cho Stable Diffusion
            # Format: [Main description], [Quality], [Lighting], [Background], [Optional modifiers]
            prompt_parts = [
                base_prompt,
                quality,
                lighting,
                background
            ]
            
            if time_weather_str:
                prompt_parts.append(time_weather_str.strip(',').strip())
            if style_str:
                prompt_parts.append(style_str.strip(',').strip())
            if composition_str:
                prompt_parts.append(composition_str.strip(',').strip())
            
            variation = ", ".join(prompt_parts)
            
            # Tránh duplicate (nếu có)
            if variation not in used_combinations or len(used_combinations) < num_variations:
                variations.append(variation)
                used_combinations.add(variation)
            else:
                # Nếu duplicate, thử lại với combination khác
                i -= 1
                continue
        
        return variations