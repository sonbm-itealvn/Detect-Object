"""
Safety Classifier - Tầng 1: Phân loại 3 trạng thái (An toàn/Nguy hiểm/Nghi ngờ)
Kết hợp với LLM Safety Analyzer (Tầng 2) và Local Rules Database (Tầng 3)
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

from RL.llm_safety_analyzer import SafetyLevel, LLMSafetyAnalyzer
from RL.local_rules_db import LocalRulesDatabase


class SafetyClassifier:
    """Phân loại relationships thành 3 trạng thái: An toàn, Nguy hiểm, Nghi ngờ"""
    
    def __init__(
        self,
        config_path: str = "RL/safety_config.json",
        llm_enabled: bool = True,
        local_rules_enabled: bool = True
    ):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Load danh sách trắng/đen
        self.white_list = self.config.get("white_list", {})
        self.black_list = self.config.get("black_list", {})
        self.gray_threshold = self.config.get("gray_list_threshold", 0.3)
        
        # Khởi tạo LLM Analyzer (Tầng 2)
        self.llm_enabled = llm_enabled and self.config.get("llm_enabled", True)
        self.llm_analyzer = None
        if self.llm_enabled:
            try:
                self.llm_analyzer = LLMSafetyAnalyzer(
                    provider=self.config.get("llm_provider", "openai"),
                    model=self.config.get("llm_model", "gpt-3.5-turbo"),
                    enabled=True
                )
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize LLM analyzer: {e}")
                self.llm_enabled = False
        
        # Khởi tạo Local Rules Database (Tầng 3)
        self.local_rules_enabled = local_rules_enabled
        self.local_rules_db = None
        if self.local_rules_enabled:
            try:
                self.local_rules_db = LocalRulesDatabase()
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize Local Rules DB: {e}")
                self.local_rules_enabled = False
        
        # Alert levels
        self.alert_levels = self.config.get("alert_levels", {})
    
    def _load_config(self) -> Dict:
        """Load cấu hình từ file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading config: {e}")
                return {}
        return {}
    
    def _normalize_text(self, text: str) -> str:
        """Chuẩn hóa text để so sánh"""
        return text.lower().strip()
    
    def _match_pattern(
        self,
        subject: str,
        relation: str,
        object_name: str,
        pattern: Dict[str, str]
    ) -> bool:
        """Kiểm tra xem relationship có khớp với pattern không"""
        subj_pattern = pattern.get("subject_pattern", "")
        rel_pattern = pattern.get("relation_pattern", "")
        obj_pattern = pattern.get("object_pattern", "")
        
        subj_match = re.search(subj_pattern, self._normalize_text(subject), re.IGNORECASE) if subj_pattern else True
        rel_match = re.search(rel_pattern, self._normalize_text(relation), re.IGNORECASE) if rel_pattern else True
        obj_match = re.search(obj_pattern, self._normalize_text(object_name), re.IGNORECASE) if obj_pattern else True
        
        return subj_match and rel_match and obj_match
    
    def _check_white_list(
        self,
        subject: str,
        relation: str,
        object_name: str
    ) -> bool:
        """Kiểm tra xem relationship có trong danh sách trắng không"""
        # Kiểm tra exact match
        relationships = self.white_list.get("relationships", [])
        for rel in relationships:
            if (self._normalize_text(rel.get("subject", "")) == self._normalize_text(subject) and
                self._normalize_text(rel.get("relation", "")) == self._normalize_text(relation) and
                self._normalize_text(rel.get("object", "")) == self._normalize_text(object_name)):
                return True
        
        # Kiểm tra pattern match
        patterns = self.white_list.get("patterns", [])
        for pattern in patterns:
            if self._match_pattern(subject, relation, object_name, pattern):
                return True
        
        return False
    
    def _check_black_list(
        self,
        subject: str,
        relation: str,
        object_name: str
    ) -> bool:
        """Kiểm tra xem relationship có trong danh sách đen không"""
        # Kiểm tra exact match
        relationships = self.black_list.get("relationships", [])
        for rel in relationships:
            if (self._normalize_text(rel.get("subject", "")) == self._normalize_text(subject) and
                self._normalize_text(rel.get("relation", "")) == self._normalize_text(relation) and
                self._normalize_text(rel.get("object", "")) == self._normalize_text(object_name)):
                return True
        
        # Kiểm tra pattern match
        patterns = self.black_list.get("patterns", [])
        for pattern in patterns:
            if self._match_pattern(subject, relation, object_name, pattern):
                return True
        
        return False
    
    def _check_local_rules(
        self,
        subject: str,
        relation: str,
        object_name: str
    ) -> Optional[SafetyLevel]:
        """Kiểm tra quy tắc cục bộ (Tầng 3)"""
        if not self.local_rules_enabled or not self.local_rules_db:
            return None
        
        rule = self.local_rules_db.get_rule(subject, relation, object_name)
        if rule:
            level_str = rule.get("level", "suspicious")
            return SafetyLevel(level_str)
        return None
    
    def classify(
        self,
        subject: str,
        relation: str,
        object_name: str,
        confidence: float = 1.0
    ) -> Tuple[SafetyLevel, float, str]:
        """
        Phân loại relationship thành 3 trạng thái
        
        Returns:
            (SafetyLevel, confidence, explanation)
        """
        # Tầng 3: Kiểm tra quy tắc cục bộ trước (ưu tiên cao nhất)
        local_rule = self._check_local_rules(subject, relation, object_name)
        if local_rule:
            return local_rule, 1.0, "Local rule"
        
        # Tầng 1: Kiểm tra danh sách đen (nguy hiểm rõ ràng)
        if self._check_black_list(subject, relation, object_name):
            return SafetyLevel.DANGEROUS, 0.95, "Black list match"
        
        # Tầng 1: Kiểm tra danh sách trắng (an toàn)
        if self._check_white_list(subject, relation, object_name):
            return SafetyLevel.SAFE, 0.9, "White list match"
        
        # Tầng 2: Nếu không có trong danh sách, hỏi LLM
        if self.llm_enabled and self.llm_analyzer:
            try:
                llm_level, llm_conf, llm_exp = self.llm_analyzer.analyze_relationship(
                    subject, relation, object_name
                )
                return llm_level, llm_conf * confidence, f"LLM: {llm_exp}"
            except Exception as e:
                print(f"⚠️ LLM analysis error: {e}")
        
        # Mặc định: Nghi ngờ (Gray-list) - "Cái gì chưa biết thì phải Cảnh giác"
        return SafetyLevel.SUSPICIOUS, self.gray_threshold, "Unknown relationship - defaulting to suspicious"
    
    def classify_batch(
        self,
        relationships: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Phân loại nhiều relationships cùng lúc"""
        results = []
        for rel in relationships:
            subject = rel.get("subject", "")
            relation = rel.get("relation", "")
            object_name = rel.get("object", "")
            confidence = rel.get("confidence", 1.0)
            
            level, conf, explanation = self.classify(subject, relation, object_name, confidence)
            
            result = {
                **rel,
                "safety_level": level.value,
                "safety_confidence": conf,
                "safety_explanation": explanation
            }
            results.append(result)
        
        return results
    
    def get_alert_info(self, level: SafetyLevel) -> Dict:
        """Lấy thông tin cảnh báo cho level"""
        level_str = level.value
        return self.alert_levels.get(level_str, {
            "color": [128, 128, 128],
            "message": "Unknown",
            "sound_alert": False
        })


if __name__ == "__main__":
    # Test
    classifier = SafetyClassifier()
    
    test_cases = [
        {"subject": "person", "relation": "sitting on", "object": "chair"},
        {"subject": "child", "relation": "eating", "object": "battery"},
        {"subject": "person", "relation": "holding", "object": "knife"},
        {"subject": "child", "relation": "putting in mouth", "object": "coin"},
    ]
    
    for case in test_cases:
        level, conf, exp = classifier.classify(
            case["subject"],
            case["relation"],
            case["object"]
        )
        alert_info = classifier.get_alert_info(level)
        print(f"{case} -> {level.value} (conf: {conf:.2f}) - {alert_info['message']}")

