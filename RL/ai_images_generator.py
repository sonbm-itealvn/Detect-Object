# File: ai_image_generator.py
import torch
import json
import random
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

# Optional dependencies
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# Try to import diffusers, fallback to mock if not available
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("WARNING: diffusers not available, using mock generator")
    DIFFUSERS_AVAILABLE = False


class ImageQualityFilter:
    """
    Lightweight quality filter to reject low-quality AI-generated images.
    Checks: size/aspect, blur, exposure, duplicate (pHash), CLIP similarity.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.cfg = {
            "min_width": 512,
            "min_height": 512,
            "max_aspect": 2.2,
            "blur_var_threshold": 60.0,
            "exposure_min_mean": 20.0,
            "exposure_max_mean": 235.0,
            "exposure_clip_ratio": 0.20,
            "clip_min_similarity": 0.23,
            "enable_clip": CLIP_AVAILABLE,
            "enable_dupe": IMAGEHASH_AVAILABLE,
        }
        self.hash_store = set()
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()

    def _check_size_aspect(self, img: Image.Image) -> Tuple[bool, str]:
        w, h = img.size
        if w < self.cfg["min_width"] or h < self.cfg["min_height"]:
            return False, "too_small"
        ar = max(w, h) / max(1, min(w, h))
        if ar > self.cfg["max_aspect"]:
            return False, "bad_aspect"
        return True, ""

    def _check_blur(self, img: Image.Image) -> Tuple[bool, str]:
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        var = cv2.Laplacian(g, cv2.CV_64F).var()
        return var >= self.cfg["blur_var_threshold"], f"blur_var={var:.1f}"

    def _check_exposure(self, img: Image.Image) -> Tuple[bool, str]:
        g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        mean = g.mean()
        clip_ratio = ((g == 0).sum() + (g == 255).sum()) / g.size
        ok = (
            self.cfg["exposure_min_mean"] <= mean <= self.cfg["exposure_max_mean"]
            and clip_ratio <= self.cfg["exposure_clip_ratio"]
        )
        return ok, f"mean={mean:.1f},clip={clip_ratio:.3f}"

    def _check_dupe(self, img: Image.Image) -> Tuple[bool, str]:
        if not self.cfg["enable_dupe"]:
            return True, ""
        h = imagehash.phash(img)
        if str(h) in self.hash_store:
            return False, "duplicate"
        self.hash_store.add(str(h))
        return True, ""

    def _check_clip_similarity(self, img: Image.Image, prompt: str) -> Tuple[bool, str]:
        if not self.cfg["enable_clip"]:
            return True, ""
        with torch.no_grad():
            text = clip.tokenize([prompt]).to(self.device)
            image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            image_feat = self.clip_model.encode_image(image_input)
            text_feat = self.clip_model.encode_text(text)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            sim = (image_feat @ text_feat.T).item()
        return sim >= self.cfg["clip_min_similarity"], f"clip_sim={sim:.3f}"

    def evaluate(self, img: Image.Image, prompt: str) -> Tuple[bool, str]:
        checks = [
            self._check_size_aspect,
            self._check_blur,
            self._check_exposure,
            self._check_dupe,
            lambda im: self._check_clip_similarity(im, prompt),
        ]
        for fn in checks:
            ok, info = fn(img)
            if not ok:
                return False, info
        return True, "ok"


class RelationshipImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        print("Initializing AI Image Generator...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not DIFFUSERS_AVAILABLE:
            print("WARNING: diffusers not available, using mock generator")
            self.pipe = None
        else:
            try:
                # Check if CUDA is available
                print(f"Using device: {device}")
                
                print("Loading Stable Diffusion model... This may take a while...")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=True
                )
                self.pipe = self.pipe.to(device)
                
                # ========== SPEED OPTIMIZATIONS ==========
                if device == "cuda":
                    # Enable memory efficient attention (xFormers or native)
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                        print("SUCCESS: xFormers enabled for 2x speedup")
                    except Exception:
                        try:
                            from diffusers.models.attention_processor import AttnProcessor2_0
                            self.pipe.unet.set_attn_processor(AttnProcessor2_0())
                            print("SUCCESS: Flash Attention 2.0 enabled")
                        except Exception:
                            pass
                    
                    # Enable VAE slicing for lower memory
                    self.pipe.enable_vae_slicing()
                    
                    # Enable channels last memory format
                    self.pipe.unet.to(memory_format=torch.channels_last)
                    
                    # Compile UNet with torch.compile for extra speed (PyTorch 2.0+)
                    try:
                        if hasattr(torch, 'compile'):
                            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
                            print("SUCCESS: torch.compile enabled for optimized inference")
                    except Exception:
                        pass
                
                print("SUCCESS: Stable Diffusion model loaded with optimizations!")
            except Exception as e:
                print(f"ERROR: Failed to load Stable Diffusion model: {e}")
                print("Falling back to mock generator")
                self.pipe = None
                device = "cpu"
        
        self.relationship_templates = self.load_relationship_templates()
        
        # Default generation settings (can be adjusted for speed vs quality)
        self.default_num_inference_steps = 25  # Reduced from 20 for faster generation
        self.default_guidance_scale = 7.0
        self.quality_filter = ImageQualityFilter(device=device if DIFFUSERS_AVAILABLE else "cpu")
    
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
                passed, reason = self.quality_filter.evaluate(mock_image, prompt)
                if not passed:
                    print(f"  Skipped mock image {i+1}: {reason}")
                    continue
                generated_images.append({
                    'image': mock_image,
                    'prompt': prompt,
                    'original_relationship': relationship,
                    'is_mock': True
                })
        else:
            print(f"Generating {len(variations)} images with Stable Diffusion (optimized)...")
            
            # OPTIMIZATION: Batch generation when possible
            batch_size = min(len(variations), 2)  # Batch size based on GPU memory
            
            for batch_start in range(0, len(variations), batch_size):
                batch_prompts = variations[batch_start:batch_start + batch_size]
                batch_idx = batch_start // batch_size + 1
                total_batches = (len(variations) + batch_size - 1) // batch_size
                
                print(f"  Batch {batch_idx}/{total_batches}: generating {len(batch_prompts)} images...")
                
                try:
                    # Generate batch with optimized settings
                    with torch.inference_mode():  # Faster than torch.no_grad()
                        results = self.pipe(
                            batch_prompts,
                            num_inference_steps=self.default_num_inference_steps,
                            guidance_scale=self.default_guidance_scale,
                        )
                    
                    for img_idx, (image, prompt) in enumerate(zip(results.images, batch_prompts)):
                        passed, reason = self.quality_filter.evaluate(image, prompt)
                        if not passed:
                            print(f"  Skipped image (batch {batch_idx}, idx {img_idx}): {reason}")
                            continue
                        generated_images.append({
                            'image': image,
                            'prompt': prompt,
                            'original_relationship': relationship,
                            'is_mock': False
                        })
                        
                except Exception as e:
                    print(f"ERROR: Batch generation failed: {e}, falling back to single generation")
                    # Fallback to single image generation
                    for prompt in batch_prompts:
                        try:
                            with torch.inference_mode():
                                image = self.pipe(
                                    prompt, 
                                    num_inference_steps=self.default_num_inference_steps,
                                    guidance_scale=self.default_guidance_scale,
                                ).images[0]
                            passed, reason = self.quality_filter.evaluate(image, prompt)
                            if not passed:
                                print(f"  Skipped image (fallback): {reason}")
                                continue
                            generated_images.append({
                                'image': image,
                                'prompt': prompt,
                                'original_relationship': relationship,
                                'is_mock': False
                            })
                        except Exception as e2:
                            print(f"ERROR: Single generation also failed: {e2}")
                            mock_image = self.create_mock_image()
                            generated_images.append({
                                'image': mock_image,
                                'prompt': prompt,
                                'original_relationship': relationship,
                                'is_mock': True
                            })
                
                # Clear CUDA cache between batches to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
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