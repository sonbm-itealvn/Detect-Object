"""
Local Rules Database - Tầng 3: Human-in-the-loop
Lưu trữ quy tắc cục bộ cho từng gia đình, cập nhật qua phản hồi người dùng
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from RL.llm_safety_analyzer import SafetyLevel


class LocalRulesDatabase:
    """Quản lý quy tắc cục bộ cho từng gia đình"""
    
    def __init__(self, db_path: str = "local_safety_rules.json"):
        self.db_path = Path(db_path)
        self.rules = self._load_rules()
        self.stats = {
            "total_rules": 0,
            "user_feedback_count": 0,
            "last_updated": None
        }
    
    def _load_rules(self) -> Dict:
        """Load quy tắc từ file"""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.stats = data.get("stats", self.stats)
                    return data.get("rules", {})
            except Exception as e:
                print(f"⚠️ Error loading rules: {e}")
                return {}
        return {}
    
    def _save_rules(self):
        """Lưu quy tắc vào file"""
        try:
            data = {
                "rules": self.rules,
                "stats": self.stats,
                "last_updated": time.time()
            }
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving rules: {e}")
    
    def _normalize_key(self, subject: str, relation: str, object_name: str) -> str:
        """Tạo key chuẩn hóa cho rule"""
        return f"{subject.lower()}|{relation.lower()}|{object_name.lower()}"
    
    def add_rule(
        self,
        subject: str,
        relation: str,
        object_name: str,
        level: SafetyLevel,
        source: str = "user_feedback",
        metadata: Optional[Dict] = None
    ):
        """
        Thêm/quy tắc mới từ phản hồi người dùng
        
        Args:
            subject: Chủ thể
            relation: Hành động
            object_name: Đối tượng
            level: Mức độ an toàn
            source: Nguồn (user_feedback, admin, etc.)
            metadata: Thông tin bổ sung
        """
        key = self._normalize_key(subject, relation, object_name)
        
        rule = {
            "subject": subject,
            "relation": relation,
            "object": object_name,
            "level": level.value,
            "source": source,
            "created_at": time.time(),
            "updated_at": time.time(),
            "usage_count": 0,
            "metadata": metadata or {}
        }
        
        # Nếu rule đã tồn tại, cập nhật
        if key in self.rules:
            old_rule = self.rules[key]
            rule["created_at"] = old_rule.get("created_at", time.time())
            rule["usage_count"] = old_rule.get("usage_count", 0)
        
        self.rules[key] = rule
        self.stats["total_rules"] = len(self.rules)
        if source == "user_feedback":
            self.stats["user_feedback_count"] += 1
        self.stats["last_updated"] = time.time()
        
        self._save_rules()
        print(f"✅ Added rule: {subject} {relation} {object_name} -> {level.value}")
    
    def get_rule(
        self,
        subject: str,
        relation: str,
        object_name: str
    ) -> Optional[Dict]:
        """Lấy quy tắc cho relationship"""
        key = self._normalize_key(subject, relation, object_name)
        rule = self.rules.get(key)
        
        if rule:
            # Tăng usage count
            rule["usage_count"] = rule.get("usage_count", 0) + 1
            rule["updated_at"] = time.time()
            self._save_rules()
        
        return rule
    
    def remove_rule(
        self,
        subject: str,
        relation: str,
        object_name: str
    ) -> bool:
        """Xóa quy tắc"""
        key = self._normalize_key(subject, relation, object_name)
        if key in self.rules:
            del self.rules[key]
            self.stats["total_rules"] = len(self.rules)
            self.stats["last_updated"] = time.time()
            self._save_rules()
            return True
        return False
    
    def get_all_rules(self) -> List[Dict]:
        """Lấy tất cả quy tắc"""
        return list(self.rules.values())
    
    def get_rules_by_level(self, level: SafetyLevel) -> List[Dict]:
        """Lấy quy tắc theo mức độ"""
        return [
            rule for rule in self.rules.values()
            if rule.get("level") == level.value
        ]
    
    def get_stats(self) -> Dict:
        """Lấy thống kê"""
        return {
            **self.stats,
            "rules_by_level": {
                "safe": len(self.get_rules_by_level(SafetyLevel.SAFE)),
                "suspicious": len(self.get_rules_by_level(SafetyLevel.SUSPICIOUS)),
                "dangerous": len(self.get_rules_by_level(SafetyLevel.DANGEROUS))
            }
        }
    
    def export_rules(self, output_path: str):
        """Xuất quy tắc ra file"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "rules": self.rules,
                    "stats": self.stats
                }, f, ensure_ascii=False, indent=2)
            print(f"✅ Exported rules to {output_path}")
        except Exception as e:
            print(f"⚠️ Error exporting rules: {e}")
    
    def import_rules(self, input_path: str, merge: bool = True):
        """Nhập quy tắc từ file"""
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                imported_rules = data.get("rules", {})
                
                if merge:
                    self.rules.update(imported_rules)
                else:
                    self.rules = imported_rules
                
                self.stats["total_rules"] = len(self.rules)
                self.stats["last_updated"] = time.time()
                self._save_rules()
                print(f"✅ Imported {len(imported_rules)} rules from {input_path}")
        except Exception as e:
            print(f"⚠️ Error importing rules: {e}")


if __name__ == "__main__":
    # Test
    db = LocalRulesDatabase()
    
    # Thêm quy tắc từ phản hồi người dùng
    db.add_rule(
        "child",
        "playing with",
        "toy",
        SafetyLevel.SAFE,
        source="user_feedback",
        metadata={"user_id": "user123", "context": "normal play"}
    )
    
    db.add_rule(
        "child",
        "eating",
        "medicine",
        SafetyLevel.DANGEROUS,
        source="user_feedback"
    )
    
    # Lấy quy tắc
    rule = db.get_rule("child", "playing with", "toy")
    print(f"Rule: {rule}")
    
    # Thống kê
    stats = db.get_stats()
    print(f"Stats: {stats}")

