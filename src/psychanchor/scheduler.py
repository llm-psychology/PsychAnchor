#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基於 LLM 的智能排程系統
自動根據人物個性、需求和環境生成個性化活動安排
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import asyncio

from personadough.runtime.responder import LLM_responder
from psychanchor.persona_loader import PersonaLoader

logger = logging.getLogger(__name__)

@dataclass
class PersonalityProfile:
    """人物個性檔案"""
    persona_id: str
    name: str
    age: int
    occupation: str
    personality_traits: List[str]
    interests: List[str]
    daily_preferences: Dict[str, Any]
    energy_pattern: str  # "morning", "evening", "flexible"
    social_level: str    # "introvert", "extrovert", "ambivert"
    
@dataclass
class ContextInfo:
    """環境上下文資訊"""
    current_time: str
    weather: str
    season: str
    available_locations: List[str]
    other_characters: List[str]
    special_events: List[str]
    
@dataclass
class ActivityRequest:
    """活動生成請求"""
    persona_id: str
    time_slot: str
    duration_minutes: int
    context: ContextInfo
    constraints: List[str]
    
@dataclass
class GeneratedActivity:
    """生成的活動"""
    name: str
    activity_type: str
    location: str
    duration: int
    priority: float
    reasoning: str
    social_participants: List[str]
    required_items: List[str]

class LLMScheduler:
    """基於 LLM 的智能排程器"""
    
    def __init__(self, use_ollama=None):
        self.llm_responder = None
        self.persona_loader = None
        self.personality_cache = {}
        self.use_ollama = use_ollama
        
        # 活動類型映射
        self.activity_types = {
            "rest": "休息放鬆",
            "sleep": "睡眠休息", 
            "work": "工作學習",
            "study": "學習進修",
            "eat": "用餐飲食",
            "play_football": "運動娛樂",
            "socialize": "社交互動",
            "exercise": "體能訓練",
            "hobby": "興趣愛好",
            "shopping": "購物採買",
            "travel": "外出旅行"
        }
        
        # 位置類型
        self.location_types = {
            "home": "家中",
            "school": "學校",
            "workplace": "工作場所", 
            "restaurant": "餐廳",
            "park": "公園",
            "soccer_field": "運動場",
            "library": "圖書館",
            "shopping_mall": "購物中心",
            "cafe": "咖啡廳"
        }
        
    async def load_personality_profile(self, persona_id: str) -> PersonalityProfile:
        """載入人物個性檔案"""
        if persona_id in self.personality_cache:
            return self.personality_cache[persona_id]
            
        try:
            # 延遲初始化 persona_loader
            if self.persona_loader is None:
                self.persona_loader = PersonaLoader()
                await self.persona_loader.wait_until_ready()
            
            # 載入 persona 資料
            persona_data = self.persona_loader.get_persona_by_id(persona_id)
            if not persona_data:
                raise ValueError(f"找不到 persona {persona_id}")
            
            # 解析個性特徵
            personality_traits = self._extract_personality_traits(persona_data)
            interests = self._extract_interests(persona_data)
            
            profile = PersonalityProfile(
                persona_id=persona_id,
                name=persona_data.get('基本資料', {}).get('姓名', 'Unknown'),
                age=persona_data.get('基本資料', {}).get('年紀', 25),
                occupation=persona_data.get('occupation', 'Student'),
                personality_traits=personality_traits,
                interests=interests,
                daily_preferences=self._extract_daily_preferences(persona_data),
                energy_pattern=self._determine_energy_pattern(persona_data),
                social_level=self._determine_social_level(persona_data)
            )
            
            self.personality_cache[persona_id] = profile
            return profile
            
        except Exception as e:
            logger.error(f"載入個性檔案失敗: {e}")
            # 返回預設檔案
            return PersonalityProfile(
                persona_id=persona_id,
                name="Unknown",
                age=25,
                occupation="Student",
                personality_traits=["friendly", "curious"],
                interests=["reading", "music"],
                daily_preferences={"wake_time": "07:00", "sleep_time": "23:00"},
                energy_pattern="flexible",
                social_level="ambivert"
            )
    
    def _extract_personality_traits(self, persona_data: Dict) -> List[str]:
        """從 persona 資料中提取個性特徵"""
        traits = []
        
        # 從描述中提取關鍵詞
        description = str(persona_data.get('生平故事', '')).lower()
        personality = ' '.join(persona_data.get('人格屬性', {}).get('人格特質', [])).lower()
        
        trait_keywords = {
            'friendly': ['友善', '親切', '和善', 'friendly'],
            'outgoing': ['外向', '活潑', 'outgoing', '開朗'],
            'quiet': ['安靜', '內向', 'quiet', '沉默'],
            'creative': ['創意', '創造', 'creative', '藝術'],
            'analytical': ['分析', '理性', 'analytical', '邏輯'],
            'energetic': ['精力', '活力', 'energetic', '充滿'],
            'calm': ['冷靜', '平靜', 'calm', '穩重'],
            'curious': ['好奇', '探索', 'curious', '求知']
        }
        
        for trait, keywords in trait_keywords.items():
            if any(keyword in description or keyword in personality for keyword in keywords):
                traits.append(trait)
        
        return traits if traits else ['friendly', 'curious']
    
    def _extract_interests(self, persona_data: Dict) -> List[str]:
        """提取興趣愛好"""
        interests = []
        
        # 從各個欄位中提取興趣
        fields_to_check = ['interests', 'hobbies', 'likes', '生平故事']
        
        interest_keywords = {
            'sports': ['運動', '足球', '籃球', 'sports', 'football'],
            'reading': ['閱讀', '書籍', 'reading', '看書'],
            'music': ['音樂', '唱歌', 'music', '歌曲'],
            'cooking': ['烹飪', '做菜', 'cooking', '料理'],
            'art': ['藝術', '繪畫', 'art', '畫畫'],
            'technology': ['科技', '電腦', 'technology', '程式'],
            'travel': ['旅行', '旅遊', 'travel', '出遊'],
            'socializing': ['社交', '朋友', 'social', '聚會']
        }
        
        text_to_analyze = ' '.join([
            str(persona_data.get(field, '')) for field in fields_to_check
        ]).lower()
        
        for interest, keywords in interest_keywords.items():
            if any(keyword in text_to_analyze for keyword in keywords):
                interests.append(interest)
        
        return interests if interests else ['reading', 'music']
    
    def _extract_daily_preferences(self, persona_data: Dict) -> Dict[str, Any]:
        """提取日常偏好"""
        return {
            "wake_time": "07:00",
            "sleep_time": "23:00", 
            "meal_times": ["07:30", "12:00", "18:30"],
            "preferred_activities": ["study", "socialize", "rest"],
            "energy_peak": "morning"
        }
    
    def _determine_energy_pattern(self, persona_data: Dict) -> str:
        """判斷能量模式"""
        description = str(persona_data.get('生平故事', '')).lower()
        
        if any(word in description for word in ['早起', '晨型', 'morning']):
            return "morning"
        elif any(word in description for word in ['夜貓', '晚上', 'night', 'evening']):
            return "evening"
        else:
            return "flexible"
    
    def _determine_social_level(self, persona_data: Dict) -> str:
        """判斷社交程度"""
        description = str(persona_data.get('生平故事', '')).lower()
        personality = ' '.join(persona_data.get('人格屬性', {}).get('人格特質', [])).lower()
        
        text = f"{description} {personality}"
        
        if any(word in text for word in ['外向', '社交', '活潑', 'extrovert', 'outgoing']):
            return "extrovert"
        elif any(word in text for word in ['內向', '安靜', 'introvert', 'quiet', 'shy']):
            return "introvert"
        else:
            return "ambivert"
    
    async def generate_activity(self, request: ActivityRequest) -> GeneratedActivity:
        """使用 LLM 生成個性化活動"""
        
        # 載入人物檔案
        profile = await self.load_personality_profile(request.persona_id)
        
        # 構建 LLM 提示
        prompt = self._build_activity_prompt(profile, request)
        
        try:
            # 延遲初始化 LLM
            if self.llm_responder is None:
                try:
                    self.llm_responder = LLM_responder(use_ollama=self.use_ollama)
                except Exception as e:
                    logger.warning(f"LLM 初始化失敗，使用預設活動: {e}")
                    return self._generate_fallback_activity(profile, request)
            
            # 呼叫 LLM
            response = await self.llm_responder.full_chat_gpt_4o(
                sys_prompt="你是一個專業的生活規劃師，專門為虛擬角色安排個性化的日常活動。",
                usr_prompt=prompt
            )
            
            # 解析回應
            activity = self._parse_llm_response(response, request)
            
            logger.info(f"為 {profile.name} 生成活動: {activity.name}")
            return activity
            
        except Exception as e:
            logger.error(f"LLM 活動生成失敗: {e}")
            # 返回預設活動
            return self._generate_fallback_activity(profile, request)
    
    def _build_activity_prompt(self, profile: PersonalityProfile, request: ActivityRequest) -> str:
        """構建 LLM 提示"""
        
        prompt = f"""
你是一個專業的生活規劃師，需要為以下人物安排合適的活動：

## 人物資訊
- 姓名: {profile.name}
- 年齡: {profile.age}
- 職業: {profile.occupation}
- 個性特徵: {', '.join(profile.personality_traits)}
- 興趣愛好: {', '.join(profile.interests)}
- 能量模式: {profile.energy_pattern}
- 社交程度: {profile.social_level}

## 時間與環境
- 時間段: {request.time_slot}
- 持續時間: {request.duration_minutes} 分鐘
- 當前時間: {request.context.current_time}
- 天氣: {request.context.weather}
- 季節: {request.context.season}
- 可用地點: {', '.join(request.context.available_locations)}
- 其他角色: {', '.join(request.context.other_characters)}

## 限制條件
{chr(10).join(f'- {constraint}' for constraint in request.constraints)}

## 可選活動類型
{chr(10).join(f'- {k}: {v}' for k, v in self.activity_types.items())}

## 可選地點
{chr(10).join(f'- {k}: {v}' for k, v in self.location_types.items())}

請根據以上資訊，為這個人物設計一個合適的活動。回應必須是有效的 JSON 格式：

{{
    "name": "活動名稱",
    "activity_type": "活動類型代碼",
    "location": "地點代碼", 
    "duration": 持續時間(分鐘),
    "priority": 優先級(0.0-1.0),
    "reasoning": "選擇這個活動的理由",
    "social_participants": ["參與的其他角色ID"],
    "required_items": ["需要的物品"]
}}

請確保：
1. 活動符合人物的個性和興趣
2. 考慮時間和環境因素
3. 活動類型和地點代碼必須從上述選項中選擇
4. 提供清晰的理由說明
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, request: ActivityRequest) -> GeneratedActivity:
        """解析 LLM 回應"""
        try:
            # 提取 JSON 部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("回應中沒有找到 JSON")
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # 驗證必要欄位
            required_fields = ['name', 'activity_type', 'location', 'duration', 'priority', 'reasoning']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"缺少必要欄位: {field}")
            
            # 驗證活動類型和地點
            if data['activity_type'] not in self.activity_types:
                data['activity_type'] = 'rest'  # 預設值
                
            if data['location'] not in self.location_types:
                data['location'] = 'home'  # 預設值
            
            return GeneratedActivity(
                name=data['name'],
                activity_type=data['activity_type'],
                location=data['location'],
                duration=min(max(data['duration'], 5), request.duration_minutes),  # 限制範圍
                priority=min(max(data['priority'], 0.0), 1.0),  # 限制範圍
                reasoning=data['reasoning'],
                social_participants=data.get('social_participants', []),
                required_items=data.get('required_items', [])
            )
            
        except Exception as e:
            logger.error(f"解析 LLM 回應失敗: {e}")
            raise
    
    def _generate_fallback_activity(self, profile: PersonalityProfile, request: ActivityRequest) -> GeneratedActivity:
        """生成預設活動（當 LLM 失敗時）"""
        
        # 根據時間段選擇合適的活動
        current_hour = int(request.context.current_time.split(':')[0])
        
        if 6 <= current_hour < 9:
            activity_type = "rest"
            name = "晨間休息"
            location = "home"
        elif 9 <= current_hour < 12:
            if profile.occupation == "Student":
                activity_type = "study"
                name = "上課學習"
                location = "school"
            else:
                activity_type = "work"
                name = "工作時間"
                location = "workplace"
        elif 12 <= current_hour < 14:
            activity_type = "eat"
            name = "午餐時間"
            location = "restaurant"
        elif 14 <= current_hour < 18:
            if "sports" in profile.interests:
                activity_type = "play_football"
                name = "運動時間"
                location = "soccer_field"
            else:
                activity_type = "socialize"
                name = "社交時間"
                location = "park"
        elif 18 <= current_hour < 20:
            activity_type = "eat"
            name = "晚餐時間"
            location = "home"
        else:
            activity_type = "rest"
            name = "晚間休息"
            location = "home"
        
        return GeneratedActivity(
            name=name,
            activity_type=activity_type,
            location=location,
            duration=request.duration_minutes,
            priority=0.5,
            reasoning=f"根據 {profile.name} 的作息時間安排的預設活動",
            social_participants=[],
            required_items=[]
        )
    
    async def generate_daily_schedule(self, persona_id: str, date: str = None) -> List[GeneratedActivity]:
        """生成一整天的個性化排程"""
        
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        profile = await self.load_personality_profile(persona_id)
        activities = []
        
        # 定義時間段
        time_slots = [
            ("07:00", 60, "晨間時光"),
            ("08:00", 120, "上午活動"),
            ("10:00", 120, "上午後段"),
            ("12:00", 60, "午餐時間"),
            ("13:00", 180, "下午活動"),
            ("16:00", 120, "下午後段"),
            ("18:00", 60, "晚餐時間"),
            ("19:00", 120, "晚間活動"),
            ("21:00", 120, "夜間時光")
        ]
        
        context = ContextInfo(
            current_time=datetime.now().strftime("%H:%M"),
            weather="晴朗",
            season="春季",
            available_locations=list(self.location_types.keys()),
            other_characters=[],
            special_events=[]
        )
        
        for time_slot, duration, description in time_slots:
            request = ActivityRequest(
                persona_id=persona_id,
                time_slot=f"{time_slot} ({description})",
                duration_minutes=duration,
                context=context,
                constraints=[f"適合 {time_slot} 的活動"]
            )
            
            try:
                activity = await self.generate_activity(request)
                activities.append(activity)
                
                # 更新上下文時間
                next_time = datetime.strptime(time_slot, "%H:%M") + timedelta(minutes=duration)
                context.current_time = next_time.strftime("%H:%M")
                
            except Exception as e:
                logger.error(f"生成 {time_slot} 活動失敗: {e}")
                continue
        
        logger.info(f"為 {profile.name} 生成了 {len(activities)} 個活動")
        return activities

# 全域實例
llm_scheduler = LLMScheduler()

