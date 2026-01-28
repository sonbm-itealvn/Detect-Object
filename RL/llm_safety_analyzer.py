"""
LLM Safety Analyzer - Tầng 2: Kết hợp LLM để suy luận ngữ nghĩa
Sử dụng ChatGPT/Gemini để đánh giá mức độ nguy hiểm của relationships
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load từ file .env ở thư mục gốc
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    print("⚠️ Will try to read from environment variables only.")


class SafetyLevel(Enum):
    """3 trạng thái an toàn"""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"


class LLMSafetyAnalyzer:
    """Phân tích mức độ nguy hiểm của relationships bằng LLM"""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-5.2",
        api_key: Optional[str] = None,
        enabled: bool = True,
        cache_file: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.model = model
        self.enabled = enabled
        self.cache_file = Path(cache_file) if cache_file else Path("llm_safety_cache.json")
        self.cache = self._load_cache()
        
        # Lấy API key từ parameter, sau đó từ .env file, cuối cùng từ environment
        if not api_key:
            # Tìm file .env ở thư mục gốc project
            env_file = Path(__file__).parent.parent / ".env"
            if env_file.exists():
                try:
                    from dotenv import dotenv_values
                    env_vars = dotenv_values(env_file)
                    api_key = env_vars.get(f"{self.provider.upper()}_API_KEY") or env_vars.get("API_KEY")
                except ImportError:
                    pass
            
            # Nếu vẫn chưa có, thử từ environment variable
            if not api_key:
                api_key = os.getenv(f"{self.provider.upper()}_API_KEY") or os.getenv("API_KEY")
        
        self.api_key = api_key
        if not self.api_key and enabled:
            print(f"⚠️ Warning: {self.provider.upper()}_API_KEY not found in .env file or environment variables.")
            print(f"⚠️ Please create a .env file in the project root with: {self.provider.upper()}_API_KEY=your_key_here")
            print(f"⚠️ LLM analysis will be disabled.")
            self.enabled = False
        
        # Import provider-specific client
        self.client = None
        if self.enabled:
            self._init_client()
    
    def _init_client(self):
        """Khởi tạo client cho LLM provider"""
        try:
            if self.provider == "openai":
                try:
                    import openai
                    self.client = openai.OpenAI(api_key=self.api_key)
                except ImportError:
                    print("⚠️ openai package not installed. Install with: pip install openai")
                    self.enabled = False
            elif self.provider == "gemini":
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    self.client = genai.GenerativeModel(self.model)
                except ImportError:
                    print("⚠️ google-generativeai package not installed. Install with: pip install google-generativeai")
                    self.enabled = False
            else:
                print(f"⚠️ Unknown provider: {self.provider}")
                self.enabled = False
        except Exception as e:
            print(f"⚠️ Error initializing LLM client: {e}")
            self.enabled = False
    
    def _load_cache(self) -> Dict:
        """Load cache từ file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_cache(self):
        """Lưu cache vào file"""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving cache: {e}")
    
    def _get_cache_key(self, subject: str, relation: str, object_name: str) -> str:
        """Tạo cache key"""
        return f"{subject.lower()}|{relation.lower()}|{object_name.lower()}"
    
    def _build_prompt(self, subject: str, relation: str, object_name: str) -> str:
        """Xây dựng prompt cho LLM"""
        prompt = f"""Bạn là một hệ thống phân tích an toàn cho gia đình. Hãy đánh giá mức độ nguy hiểm của hành động sau:

Subject (Chủ thể): {subject}
Relation (Hành động): {relation}
Object (Đối tượng): {object_name}

Trong bối cảnh gia đình, đặc biệt là khi có trẻ em, hành động này có nguy hiểm không?

Hãy trả lời CHỈ bằng một trong ba từ sau (không giải thích thêm):
- "SAFE" nếu hành động hoàn toàn an toàn, bình thường
- "SUSPICIOUS" nếu hành động có vẻ kỳ lạ, không rõ ràng, cần cảnh giác
- "DANGEROUS" nếu hành động có nguy cơ gây thương tích, ngộ độc, hoặc nguy hiểm rõ ràng

Trả lời:"""
        return prompt
    
    def analyze_relationship(
        self,
        subject: str,
        relation: str,
        object_name: str,
        use_cache: bool = True
    ) -> Tuple[SafetyLevel, float, Optional[str]]:
        """
        Phân tích mức độ nguy hiểm của một relationship
        
        Returns:
            (SafetyLevel, confidence, explanation)
        """
        cache_key = self._get_cache_key(subject, relation, object_name)
        
        # Kiểm tra cache
        if use_cache and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            level_str = cached_result.get("level", "safe")
            confidence = cached_result.get("confidence", 0.8)
            explanation = cached_result.get("explanation")
            return SafetyLevel(level_str), confidence, explanation
        
        # Nếu LLM không được bật, trả về SUSPICIOUS (gray-list)
        if not self.enabled:
            return SafetyLevel.SUSPICIOUS, 0.5, "LLM analysis disabled - defaulting to suspicious"
        
        try:
            prompt = self._build_prompt(subject, relation, object_name)
            response_text = self._call_llm(prompt)
            
            # Parse response
            response_text = response_text.strip().upper()
            if "DANGEROUS" in response_text:
                level = SafetyLevel.DANGEROUS
                confidence = 0.9
            elif "SUSPICIOUS" in response_text:
                level = SafetyLevel.SUSPICIOUS
                confidence = 0.7
            else:
                level = SafetyLevel.SAFE
                confidence = 0.8
            
            explanation = response_text[:100] if len(response_text) > 100 else response_text
            
            # Lưu vào cache
            self.cache[cache_key] = {
                "level": level.value,
                "confidence": confidence,
                "explanation": explanation,
                "timestamp": time.time()
            }
            self._save_cache()
            
            return level, confidence, explanation
            
        except Exception as e:
            print(f"⚠️ Error in LLM analysis: {e}")
            # Fallback: trả về SUSPICIOUS
            return SafetyLevel.SUSPICIOUS, 0.5, f"LLM error: {str(e)}"
    
    def _call_llm(self, prompt: str) -> str:
        """Gọi LLM API"""
        if self.provider == "openai":
            # Model mới (GPT-4o, o1, etc.) dùng max_completion_tokens
            # Model cũ (GPT-3.5, GPT-4) dùng max_tokens
            # Thử max_completion_tokens trước, nếu lỗi thì fallback về max_tokens
            use_max_completion_tokens = self._should_use_max_completion_tokens()
            
            try:
                if use_max_completion_tokens:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a safety analysis system for home environments."},
                            {"role": "user", "content": prompt}
                        ],
                        max_completion_tokens=50,
                        temperature=0.3
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a safety analysis system for home environments."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=50,
                        temperature=0.3
                    )
                return response.choices[0].message.content
            except Exception as e:
                # Nếu lỗi do parameter, thử parameter khác
                error_str = str(e).lower()
                if "max_tokens" in error_str or "max_completion_tokens" in error_str:
                    try:
                        # Thử parameter ngược lại
                        if use_max_completion_tokens:
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": "You are a safety analysis system for home environments."},
                                    {"role": "user", "content": prompt}
                                ],
                                max_tokens=50,
                                temperature=0.3
                            )
                        else:
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": "You are a safety analysis system for home environments."},
                                    {"role": "user", "content": prompt}
                                ],
                                max_completion_tokens=50,
                                temperature=0.3
                            )
                        return response.choices[0].message.content
                    except Exception as e2:
                        raise e2
                else:
                    raise e
        
        elif self.provider == "gemini":
            response = self.client.generate_content(prompt)
            return response.text
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _should_use_max_completion_tokens(self) -> bool:
        """Kiểm tra xem model có cần dùng max_completion_tokens không"""
        # Các model mới của OpenAI dùng max_completion_tokens
        new_model_patterns = [
            "gpt-4o", "gpt-4o-mini", "o1", "o1-preview", "o1-mini",
            "gpt-5", "gpt-4.5"
        ]
        model_lower = self.model.lower()
        return any(pattern in model_lower for pattern in new_model_patterns)
    
    def batch_analyze(
        self,
        relationships: List[Dict[str, str]],
        use_cache: bool = True
    ) -> List[Tuple[SafetyLevel, float, Optional[str]]]:
        """Phân tích nhiều relationships cùng lúc"""
        results = []
        for rel in relationships:
            subject = rel.get("subject", "")
            relation = rel.get("relation", "")
            object_name = rel.get("object", "")
            result = self.analyze_relationship(subject, relation, object_name, use_cache)
            results.append(result)
        return results


if __name__ == "__main__":
    # Test
    analyzer = LLMSafetyAnalyzer(
        provider="openai",
        model="gpt-3.5-turbo",
        enabled=True
    )
    
    test_cases = [
        {"subject": "child", "relation": "eating", "object": "battery"},
        {"subject": "person", "relation": "sitting on", "object": "chair"},
        {"subject": "dog", "relation": "chew", "object": "electric wire"},
    ]
    
    for case in test_cases:
        level, conf, exp = analyzer.analyze_relationship(
            case["subject"],
            case["relation"],
            case["object"]
        )
        print(f"{case} -> {level.value} (confidence: {conf:.2f})")

