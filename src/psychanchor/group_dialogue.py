import os
import json
import uuid
import random
import logging
import aiofiles
from typing import List, Dict, Optional
from psychanchor.storage import history_directory
logger = logging.getLogger(__name__)

class GroupDialogueManager:
    def __init__(self):
        self.conversation_history_path = str(history_directory())
        os.makedirs(self.conversation_history_path, exist_ok=True)
        self.chatbot = None
        self._initialization_lock = None

    async def initialize(self):
        """初始化聊天處理器（線程安全）"""
        if self._initialization_lock is None:
            import asyncio
            self._initialization_lock = asyncio.Lock()
        
        async with self._initialization_lock:
            if not self.chatbot:
                logger.info("正在初始化 GroupDialogueManager 的 ChatWith 實例...")
                from psychanchor.chat import ChatWith
                self.chatbot = ChatWith()
                await self.chatbot.wait_until_ready()
                logger.info(f"GroupDialogueManager 初始化完成，載入了 {len(self.chatbot.persona_loader_obj.personas)} 個 persona")

    async def load_group_dialogue_history(self, session_id: str) -> List[Dict[str, str]]:
        """載入群組對話歷史"""
        history_file = os.path.join(self.conversation_history_path, f"group_dialogue_{session_id}.json")
        if os.path.exists(history_file):
            async with aiofiles.open(history_file, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        return []

    async def save_group_dialogue_history(self, session_id: str, history: List[Dict[str, str]]):
        """保存群組對話歷史"""
        history_file = os.path.join(self.conversation_history_path, f"group_dialogue_{session_id}.json")
        async with aiofiles.open(history_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(history, ensure_ascii=False, indent=2))

    async def create_group_session_id(self, persona_ids: List[str]) -> str:
        """創建新的群組會話 ID"""
        persona_ids_sorted = sorted(persona_ids)  # 確保順序一致
        return f"group_{'_'.join(persona_ids_sorted[:3])}_{str(uuid.uuid4())[:8]}"

    async def get_persona_context_for_group(
        self,
        persona_id: str,
        current_topic: str,
        conversation_history: List[Dict[str, str]],
        other_participants: List[str]
    ) -> tuple:
        """獲取 persona 在群組對話中的上下文和相關記憶"""
        index, docs, _ = await self.chatbot.interviewer_obj.load_rag_database(persona_id)
        persona_data = self.chatbot.persona_loader_obj.personas[persona_id]
        
        # 獲取相關記憶
        retrieved_docs, distances = await self.chatbot.interviewer_obj.retrieve_similar_docs(
            current_topic, index, docs
        )
        
        # 如果沒有足夠相關的記憶，返回空列表
        if len(distances) > 0 and all(dist > 0.7 for dist in distances):
            retrieved_docs = []
        
        # 建立群組對話上下文
        conversation_context = ""
        if conversation_history:
            conversation_context = "群組對話歷史:\n" + "\n".join([
                f"{exchange['speaker']}: {exchange['content']}" 
                for exchange in conversation_history[-8:]  # 群組對話顯示更多歷史
            ])
        
        return conversation_context, retrieved_docs

    async def generate_group_persona_response(
        self,
        persona_id: str,
        current_topic: str,
        conversation_context: str,
        retrieved_docs: List[str],
        other_participants: List[str],
        is_first_round: bool = False
    ) -> str:
        """生成 persona 在群組中的回應"""
        persona_data = self.chatbot.persona_loader_obj.personas[persona_id]
        
        participants_info = ", ".join(other_participants)
        
        group_instruction = ""
        if is_first_round:
            group_instruction = "這是群組對話的開始，你可以向大家介紹自己或回應話題。"
        else:
            group_instruction = "這是群組對話，請根據前面的對話內容自然地回應，可以對其他人的發言作出反應或提出新的話題。"
        
        from psychanchor.prompts import get_user_prompt
        
        system_content = get_user_prompt("group_dialogue_system",
                                       simplified_behavior=persona_data['簡化行為'],
                                       participants_info=participants_info,
                                       memory_context="\n".join(retrieved_docs) if retrieved_docs else "（無可參考的記憶資料）",
                                       conversation_context=conversation_context,
                                       group_instruction=group_instruction)
        
        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": current_topic}
        ]
        
        return await self.chatbot.interviewer_obj.simulate_persona_answer(prompt)

    def determine_speaking_order(self, persona_ids: List[str], speaking_order: Optional[List[str]] = None) -> List[str]:
        """決定發言順序"""
        if speaking_order and len(speaking_order) == len(persona_ids):
            # 檢查指定的順序是否包含所有參與者
            if set(speaking_order) == set(persona_ids):
                return speaking_order
        
        # 如果沒有指定順序或順序不正確，則使用原始順序
        return persona_ids

    async def determine_who_wants_to_speak(
        self,
        persona_ids: List[str],
        current_topic: str,
        recent_dialogue: List[Dict[str, str]],
        persona_data_map: Dict,
        round_num: int
    ) -> List[str]:
        """決定誰想要在這一輪發言"""
        potential_speakers = []
        
        for persona_id in persona_ids:
            # 獲取角色最近的發言時間（避免同一人連續發言太多次）
            recent_speaker_count = sum(1 for d in recent_dialogue[-3:] if d.get('speaker_id') == persona_id)
            
            # 如果最近發言太多次，降低發言意願
            if recent_speaker_count >= 2:
                continue
                
            # 第一輪確保每個人至少說一句話
            if round_num == 1:
                has_spoken = any(d.get('speaker_id') == persona_id for d in recent_dialogue)
                if not has_spoken:
                    potential_speakers.append(persona_id)
                    continue
            
            # 根據角色個性和話題相關性決定發言意願
            conv_context, retrieved_docs = await self.get_persona_context_for_group(
                persona_id, current_topic, recent_dialogue, 
                [persona_data_map[pid]['基本資料']['姓名'] for pid in persona_ids if pid != persona_id]
            )
            
            # 使用AI判斷是否想要發言
            wants_to_speak = await self.check_speaking_desire(
                persona_id, current_topic, conv_context, retrieved_docs, persona_data_map[persona_id]
            )
            
            if wants_to_speak:
                potential_speakers.append(persona_id)
        
        # 如果沒有人想發言，隨機選一個人
        if not potential_speakers and persona_ids:
            import random
            potential_speakers = [random.choice(persona_ids)]
            
        return potential_speakers

    async def check_speaking_desire(
        self,
        persona_id: str,
        current_topic: str,
        conversation_context: str,
        retrieved_docs: List[str],
        persona_data: Dict
    ) -> bool:
        """檢查角色是否想要對當前話題發言"""
        
        from psychanchor.prompts import get_system_prompt, get_user_prompt
        
        system_content = get_system_prompt("speaking_desire_check",
                                         simplified_behavior=persona_data['簡化行為'],
                                         memory_context="\n".join(retrieved_docs) if retrieved_docs else "（無可參考的記憶資料）",
                                         conversation_context=conversation_context)
        
        user_content = get_user_prompt("speaking_desire_check", current_topic=current_topic)
        
        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = await self.chatbot.interviewer_obj.simulate_persona_answer(prompt)
            # 判斷回應是否包含肯定的意思
            return "是" in response or "想" in response or "要" in response or "願意" in response
        except:
            # 如果AI判斷失敗，隨機決定（30%機率發言）
            import random
            return random.random() < 0.3

    async def conduct_group_dialogue(
        self,
        persona_ids: List[str],
        num_rounds: int,
        initial_topic: str,
        session_id: Optional[str] = None
    ) -> tuple:
        """執行群組對話 - 使用動態對話池概念"""
        await self.initialize()
        
        if len(persona_ids) < 2:
            raise ValueError("群組對話至少需要 2 個參與者")
        
        if len(persona_ids) > 10:
            raise ValueError("群組對話最多支援 10 個參與者")
        
        # 獲取或創建會話 ID
        if not session_id:
            session_id = await self.create_group_session_id(persona_ids)
            logger.info(f"創建新的群組對話會話: {session_id}")
            history = []
        else:
            history = await self.load_group_dialogue_history(session_id)
            logger.info(f"載入群組會話 {session_id} 的 {len(history)} 條歷史對話")
        
        # 獲取參與者資訊
        participants = []
        persona_data_map = {}
        for persona_id in persona_ids:
            persona_data = self.chatbot.persona_loader_obj.personas[persona_id]
            participants.append({
                "id": persona_id,
                "name": persona_data['基本資料']['姓名']
            })
            persona_data_map[persona_id] = persona_data
        
        dialogue_rounds = []
        current_topic = initial_topic
        
        # 對話池概念：每一輪可能有多人發言，也可能只有一人發言
        for round_num in range(1, num_rounds + 1):
            logger.info(f"開始第 {round_num} 輪群組對話")
            
            round_speakers = 0
            max_speakers_per_round = min(len(persona_ids), 3)  # 每輪最多3人發言
            
            while round_speakers < max_speakers_per_round:
                # 決定這一輪誰想要發言
                potential_speakers = await self.determine_who_wants_to_speak(
                    persona_ids, current_topic, history + dialogue_rounds, persona_data_map, round_num
                )
                
                if not potential_speakers:
                    break
                
                # 從想發言的人中選擇一個（可以加入優先級邏輯）
                import random
                speaker_id = random.choice(potential_speakers)
                
                # 獲取其他參與者的名字
                other_participants = [
                    persona_data_map[pid]['基本資料']['姓名'] 
                    for pid in persona_ids if pid != speaker_id
                ]
                
                # 獲取對話上下文
                conv_context, retrieved_docs = await self.get_persona_context_for_group(
                    speaker_id, 
                    current_topic, 
                    history + dialogue_rounds,
                    other_participants
                )
                
                # 生成回應
                is_first_round = (round_num == 1 and len(dialogue_rounds) == 0)
                response = await self.generate_group_persona_response(
                    speaker_id, 
                    current_topic, 
                    conv_context, 
                    retrieved_docs,
                    other_participants,
                    is_first_round
                )
                
                dialogue_rounds.append({
                    "speaker": persona_data_map[speaker_id]['基本資料']['姓名'],
                    "speaker_id": speaker_id,
                    "content": response,
                    "round": str(round_num)
                })
                
                # 更新話題為最新的回應
                current_topic = response
                round_speakers += 1
                
                logger.info(f"第{round_num}輪 - {persona_data_map[speaker_id]['基本資料']['姓名']}: {response[:50]}...")
                
                # 如果這輪已經有人發言，有機率結束這輪（讓對話更自然）
                if round_speakers >= 1 and random.random() < 0.4:  # 40%機率結束這輪
                    break
        
        # 保存對話歷史
        history.extend(dialogue_rounds)
        await self.save_group_dialogue_history(session_id, history)
        
        return dialogue_rounds, session_id, participants
