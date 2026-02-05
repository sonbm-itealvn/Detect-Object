# File: RL/llm_relationship_predictor.py
"""
LLM-Enhanced Open-Vocabulary Relationship Predictor.

Sử dụng OpenAI GPT-4 Vision để dự đoán relationships khi:
1. RelTR có confidence thấp
2. Relationship nằm ngoài 51 predefined classes
3. Cần xử lý non-contact relationships (looking at, watching)

Author: Auto-generated for DATN project
"""

import os
import io
import base64
import json
from PIL import Image
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import visual features module
from .visual_features import (
    extract_all_visual_features,
    compute_gaze_attention_vector,
    compute_spatial_description,
)

# OpenAI import
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("[LLMPredictor] ⚠️ OpenAI not installed. Run: pip install openai")


# RelTR's 51 predefined relationship classes
RELTR_CLASSES = {
    'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
    'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating',
    'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding',
    'in', 'in front of', 'laying on', 'looking at', 'lying on', 'made of',
    'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on',
    'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on',
    'standing on', 'to', 'under', 'using', 'walking in', 'walking on',
    'watching', 'wearing', 'wears', 'with'
}


class LLMRelationshipPredictor:
    """
    Open-vocabulary relationship prediction using OpenAI GPT-4 Vision.
    
    Workflow:
    1. Nhận visual features từ visual_features.py
    2. Construct prompt với spatial descriptions và gaze info
    3. Query GPT-4 Vision với union crop image
    4. Parse và validate response
    
    Usage:
        predictor = LLMRelationshipPredictor()
        result = predictor.predict(
            image, subject_bbox, object_bbox, 
            subject_class, object_class,
            reltr_prediction={"relation": "near", "confidence": 0.2}
        )
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",  # Cost-effective vision model
        confidence_threshold: float = 0.4,
        max_tokens: int = 50,
        temperature: float = 0.3,
    ):
        """
        Initialize LLM Relationship Predictor.
        
        Args:
            model: OpenAI model to use (gpt-4o-mini, gpt-4o, gpt-4-turbo)
            confidence_threshold: RelTR confidence below which to use LLM
            max_tokens: Max tokens for LLM response
            temperature: LLM temperature (lower = more deterministic)
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.client = None
        self.is_available = False
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key from environment."""
        if not OPENAI_AVAILABLE:
            print("[LLMPredictor] ❌ OpenAI package not available")
            return
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[LLMPredictor] ❌ OPENAI_API_KEY not set in environment")
            print("[LLMPredictor] 💡 Add to .env: OPENAI_API_KEY=sk-your-key-here")
            return
        
        try:
            self.client = OpenAI(api_key=api_key)
            self.is_available = True
            print(f"[LLMPredictor] ✅ Initialized with model: {self.model}")
        except Exception as e:
            print(f"[LLMPredictor] ❌ Failed to initialize OpenAI client: {e}")
    
    def _encode_image_base64(self, image: Image.Image, quality: int = 85) -> str:
        """Convert PIL Image to base64 string for API."""
        buffer = io.BytesIO()
        # Resize if too large (save tokens/cost)
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        image.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _build_prompt(
        self,
        subject: str,
        object_name: str,
        gaze_info: Dict[str, Any],
        spatial_desc: str,
        reltr_suggestion: Optional[str] = None,
    ) -> str:
        """
        Build comprehensive prompt for GPT-4 Vision.
        
        Incorporates:
        - Subject/Object class names
        - Spatial relationship description
        - Gaze analysis for non-contact relationships
        - RelTR suggestion (if available)
        """
        # Build gaze analysis section
        gaze_section = ""
        if gaze_info.get("is_animate_subject"):
            if gaze_info.get("non_contact_likelihood", 0) > 0.5:
                gaze_section = f"""
## Gaze Analysis (Important for non-contact relationships):
- Direction from subject to object: {gaze_info.get('angle_degrees', 0):.1f}°
- No physical overlap detected
- Non-contact likelihood: {gaze_info.get('non_contact_likelihood', 0):.1%}
- Consider relationships like: "looking at", "watching", "facing", "approaching"
"""
            elif gaze_info.get("has_physical_contact"):
                gaze_section = f"""
## Contact Analysis:
- Physical overlap detected (overlap ratio: {gaze_info.get('overlap_ratio', 0):.1%})
- Contact-based relationships are likely
- Consider: "holding", "carrying", "riding", "wearing", "touching"
"""
        
        # RelTR suggestion section (if low confidence)
        reltr_section = ""
        if reltr_suggestion:
            reltr_section = f"""
## Previous Model Suggestion (low confidence):
- RelTR predicted: "{reltr_suggestion}"
- You may agree or provide a more accurate relationship
"""
        
        prompt = f"""Analyze this image showing two objects and determine their most accurate relationship.

## Objects in Image:
- **Subject** (the active entity): {subject}
- **Object** (the entity being related to): {object_name}

## Spatial Information:
{spatial_desc}
{gaze_section}
{reltr_section}
## Important Guidelines:
1. Focus on what is ACTUALLY visible in the image
2. Choose the most specific relationship that applies
3. For animate subjects (person, animal), consider both contact and non-contact relationships
4. Non-contact examples: "looking at", "watching", "facing", "approaching", "following"
5. Contact examples: "holding", "carrying", "riding", "wearing", "sitting on", "standing on"
6. Spatial examples: "on", "under", "next to", "behind", "in front of", "above", "below"

## Output Format:
Respond with ONLY the relationship predicate (1-3 words).
Do NOT include any explanation or punctuation.

Examples of valid outputs:
- riding
- looking at
- next to
- playing with
- sitting on

Relationship:"""

        return prompt
    
    def _parse_response(self, response_text: str) -> str:
        """
        Parse and clean LLM response.
        
        Handles:
        - Extra whitespace
        - Quotes
        - Punctuation
        - Multi-word responses
        """
        # Clean response
        relation = response_text.strip().lower()
        
        # Remove quotes if present
        relation = relation.strip('"\'')
        
        # Remove trailing punctuation
        relation = relation.rstrip('.,!?;:')
        
        # Remove common prefixes
        prefixes = ["the relationship is", "relationship:", "answer:", "output:"]
        for prefix in prefixes:
            if relation.startswith(prefix):
                relation = relation[len(prefix):].strip()
        
        # Validate: should be 1-4 words
        words = relation.split()
        if len(words) > 4:
            # Take first 3 words as relationship
            relation = " ".join(words[:3])
        
        # Empty check
        if not relation:
            relation = "near"  # Default fallback
        
        return relation
    
    def _is_valid_relationship(self, relation: str) -> bool:
        """Check if predicted relationship is reasonable."""
        # Minimum length
        if len(relation) < 2:
            return False
        
        # Maximum words
        if len(relation.split()) > 4:
            return False
        
        # No numbers
        if any(c.isdigit() for c in relation):
            return False
        
        return True
    
    def predict(
        self,
        image: np.ndarray,
        subject_bbox: List[int],
        object_bbox: List[int],
        subject_class: str,
        object_class: str,
        reltr_prediction: Optional[Dict[str, Any]] = None,
        global_context: Optional[List[float]] = None,
        force_llm: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict relationship using GPT-4 Vision.
        
        Args:
            image: Full image (BGR numpy array)
            subject_bbox: [x1, y1, x2, y2] of subject
            object_bbox: [x1, y1, x2, y2] of object
            subject_class: Class name of subject
            object_class: Class name of object
            reltr_prediction: Optional RelTR prediction {"relation": str, "confidence": float}
            global_context: Optional scene context vector (for future use)
            force_llm: Force LLM prediction even if RelTR is confident
        
        Returns:
            Dict with:
                - relation: predicted relationship string
                - confidence: confidence score (0-1)
                - source: "llm" or "reltr"
                - llm_used: bool
                - is_open_vocab: bool (True if outside RelTR's 51 classes)
        """
        # Check if we should use LLM
        reltr_conf = reltr_prediction.get('confidence', 0) if reltr_prediction else 0
        reltr_rel = reltr_prediction.get('relation', '').lower() if reltr_prediction else ''
        
        should_use_llm = force_llm or (reltr_conf < self.confidence_threshold)
        
        if not should_use_llm:
            # Trust RelTR prediction
            return {
                'relation': reltr_rel,
                'confidence': reltr_conf,
                'source': 'reltr',
                'llm_used': False,
                'is_open_vocab': False,
            }
        
        # Check if LLM is available
        if not self.is_available:
            print("[LLMPredictor] LLM not available, returning RelTR fallback")
            return {
                'relation': reltr_rel or 'near',
                'confidence': max(reltr_conf, 0.3),
                'source': 'reltr_fallback',
                'llm_used': False,
                'is_open_vocab': False,
            }
        
        # Extract visual features
        try:
            features = extract_all_visual_features(
                image, subject_bbox, object_bbox,
                subject_class, object_class
            )
        except Exception as e:
            print(f"[LLMPredictor] Feature extraction failed: {e}")
            return {
                'relation': reltr_rel or 'near',
                'confidence': 0.3,
                'source': 'error_fallback',
                'llm_used': False,
                'is_open_vocab': False,
            }
        
        # Build prompt
        prompt = self._build_prompt(
            subject=subject_class,
            object_name=object_class,
            gaze_info=features['gaze_info'],
            spatial_desc=features['spatial_description'],
            reltr_suggestion=reltr_rel if reltr_conf > 0.1 else None,
        )
        
        # Encode image
        union_crop = features['union_crop']
        base64_image = self._encode_image_base64(union_crop)
        
        # Query GPT-4 Vision
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # Save tokens
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            raw_response = response.choices[0].message.content
            predicted_relation = self._parse_response(raw_response)
            
            # Validate
            if not self._is_valid_relationship(predicted_relation):
                print(f"[LLMPredictor] Invalid response '{raw_response}', using fallback")
                predicted_relation = reltr_rel or 'near'
            
            # Check if it's open vocabulary (outside RelTR's 51 classes)
            is_open_vocab = predicted_relation not in RELTR_CLASSES
            
            # Assign confidence
            # LLM predictions get moderate-high confidence
            llm_confidence = 0.75 if not is_open_vocab else 0.65
            
            print(f"[LLMPredictor] Predicted: '{predicted_relation}' "
                  f"(open_vocab={is_open_vocab}, RelTR suggested: '{reltr_rel}')")
            
            return {
                'relation': predicted_relation,
                'confidence': llm_confidence,
                'source': 'llm',
                'llm_used': True,
                'is_open_vocab': is_open_vocab,
                'raw_response': raw_response,
                'reltr_suggestion': reltr_rel,
                'gaze_info': features['gaze_info'],
            }
            
        except Exception as e:
            print(f"[LLMPredictor] API call failed: {e}")
            return {
                'relation': reltr_rel or 'near',
                'confidence': max(reltr_conf, 0.3),
                'source': 'api_error_fallback',
                'llm_used': False,
                'is_open_vocab': False,
                'error': str(e),
            }
    
    def predict_batch(
        self,
        image: np.ndarray,
        pairs: List[Dict[str, Any]],
        global_context: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict relationships for multiple subject-object pairs.
        
        Args:
            image: Full image (BGR numpy array)
            pairs: List of dicts with keys:
                - subject_bbox, object_bbox
                - subject_class, object_class
                - reltr_prediction (optional)
            global_context: Optional scene context
        
        Returns:
            List of prediction results
        """
        results = []
        for pair in pairs:
            result = self.predict(
                image=image,
                subject_bbox=pair['subject_bbox'],
                object_bbox=pair['object_bbox'],
                subject_class=pair['subject_class'],
                object_class=pair['object_class'],
                reltr_prediction=pair.get('reltr_prediction'),
                global_context=global_context,
            )
            results.append(result)
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Return predictor status."""
        return {
            'is_available': self.is_available,
            'model': self.model,
            'confidence_threshold': self.confidence_threshold,
            'openai_available': OPENAI_AVAILABLE,
            'api_key_set': bool(os.getenv("OPENAI_API_KEY")),
        }
    
    def predict_for_missing_pair(
        self,
        image: np.ndarray,
        subject_bbox: List[int],
        object_bbox: List[int],
        subject_class: str,
        object_class: str,
        global_context: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Predict relationship for ANY pair that RelTR did NOT detect any relation.
        
        This is a GENERALIZED method - works for ALL object pair types, not just
        specific categories like person-vehicle. It:
        1. Forces LLM to analyze the pair (no reltr_prediction)
        2. Filters out meaningless/vague relations
        3. Returns None if no meaningful relation is found
        
        Args:
            image: Full image (BGR numpy array)
            subject_bbox: [x1, y1, x2, y2] of subject
            object_bbox: [x1, y1, x2, y2] of object
            subject_class: Class name of subject
            object_class: Class name of object
            global_context: Optional scene context vector
        
        Returns:
            Dict with relation info if meaningful relation found, None otherwise
        """
        # Relations to REJECT (too vague, non-specific, or non-actionable)
        # These apply to ALL object pair types
        REJECT_RELATIONS = {
            # Too vague/generic
            'near', 'with', 'and', 'of', 'at', 'to', 'for', 'from', 'by',
            # No relation indicators
            'no relation', 'none', 'unknown', 'unclear', 'unrelated',
            'no relationship', 'nothing', 'n/a', 'na',
        }
        
        if not self.is_available:
            print("[LLMPredictor] LLM not available for missing pair prediction")
            return None
        
        # Force LLM prediction (no reltr_prediction means low confidence path)
        result = self.predict(
            image=image,
            subject_bbox=subject_bbox,
            object_bbox=object_bbox,
            subject_class=subject_class,
            object_class=object_class,
            reltr_prediction=None,  # No existing prediction
            global_context=global_context,
            force_llm=True,  # Force LLM to analyze
        )
        
        if not result.get('llm_used'):
            return None
        
        predicted_relation = result.get('relation', '').lower().strip()
        
        # Reject meaningless/vague relations
        if predicted_relation in REJECT_RELATIONS:
            print(f"[LLMPredictor] Rejecting vague relation '{predicted_relation}' for {subject_class}-{object_class}")
            return None
        
        # Accept if relation has substance (length > 2 and not in reject list)
        # This allows open-vocabulary relations for any object pair
        if len(predicted_relation) <= 2:
            print(f"[LLMPredictor] Rejecting too-short relation '{predicted_relation}' for {subject_class}-{object_class}")
            return None
        
        print(f"🤖 [LLMPredictor] Found meaningful relation: {subject_class} '{predicted_relation}' {object_class}")
        
        return {
            'relation': result['relation'],
            'confidence': result['confidence'],
            'source': 'llm_missing_pair',
            'is_open_vocab': result.get('is_open_vocab', False),
            'llm_used': True,
        }


# Singleton instance for easy access
_predictor_instance: Optional[LLMRelationshipPredictor] = None


def get_llm_predictor(force_reinit: bool = False, **kwargs) -> LLMRelationshipPredictor:
    """Get or create LLM predictor singleton."""
    global _predictor_instance
    if _predictor_instance is None or force_reinit:
        _predictor_instance = LLMRelationshipPredictor(**kwargs)
    return _predictor_instance


# ============ TESTING ============
if __name__ == "__main__":
    import cv2
    
    print("Testing LLM Relationship Predictor...")
    print("=" * 50)
    
    # Check status
    predictor = get_llm_predictor()
    status = predictor.get_status()
    print(f"Status: {json.dumps(status, indent=2)}")
    
    if not status['is_available']:
        print("\n⚠️ LLM not available. Make sure OPENAI_API_KEY is set in .env")
        print("Example .env content:")
        print("  OPENAI_API_KEY=sk-your-api-key-here")
    else:
        # Create test image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(dummy_image, (100, 100), (200, 200), (255, 0, 0), -1)
        cv2.rectangle(dummy_image, (250, 150), (350, 250), (0, 255, 0), -1)
        
        # Test prediction
        result = predictor.predict(
            image=dummy_image,
            subject_bbox=[100, 100, 200, 200],
            object_bbox=[250, 150, 350, 250],
            subject_class="person",
            object_class="car",
            reltr_prediction={"relation": "near", "confidence": 0.3},
            force_llm=True,
        )
        
        print(f"\nPrediction result:")
        print(json.dumps({k: v for k, v in result.items() if k != 'gaze_info'}, indent=2))
        print("\n✅ Test completed!")
