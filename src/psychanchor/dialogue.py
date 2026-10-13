from psychanchor.storage import rag_directory, history_directory
import logging
import json
import os
import uuid
from typing import List, Dict, Optional, Tuple
import aiofiles

from .chat import ChatWith
from .interviewer import generate_persona_rag_by_id

logger = logging.getLogger(__name__)

class PersonaDialogueManager:
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
                logger.info("正在初始化 PersonaDialogueManager 的 ChatWith 實例...")
                self.chatbot = ChatWith()
                await self.chatbot.wait_until_ready()
                logger.info(f"PersonaDialogueManager 初始化完成，載入了 {len(self.chatbot.persona_loader_obj.personas)} 個 persona")

    async def load_dialogue_history(self, session_id: str) -> List[Dict[str, str]]:
        """載入對話歷史"""
        history_file = os.path.join(self.conversation_history_path, f"dialogue_{session_id}.json")
        if os.path.exists(history_file):
            async with aiofiles.open(history_file, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        return []

    async def save_dialogue_history(self, session_id: str, history: List[Dict[str, str]]):
        """保存對話歷史"""
        history_file = os.path.join(self.conversation_history_path, f"dialogue_{session_id}.json")
        async with aiofiles.open(history_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(history, ensure_ascii=False, indent=2))

    async def create_session_id(self, persona1_id: str, persona2_id: str, is_unity: bool = False) -> str:
        """創建新的會話 ID"""
        if is_unity:
            return f"dialogue_{persona1_id}_{persona2_id}_unity"
        else:
            return f"dialogue_{persona1_id}_{persona2_id}_{str(uuid.uuid4())[:8]}"

    async def get_persona_context(
        self,
        persona_id: str,
        current_topic: str,
        conversation_history: List[Dict[str, str]],
        other_persona_name: str,
        unity_memory_context: str = ""
    ) -> Tuple[str, List[str]]:
        """獲取 persona 的對話上下文和相關記憶"""
        index, docs, _ = await self.chatbot.interviewer_obj.load_rag_database(persona_id)
        persona_data = self.chatbot.persona_loader_obj.personas[persona_id]
        
        # 獲取相關記憶
        retrieved_docs, distances = await self.chatbot.interviewer_obj.retrieve_similar_docs(
            current_topic, index, docs
        )
        
        # 如果沒有足夠相關的記憶，返回空列表
        if len(distances) > 0 and all(dist > 0.7 for dist in distances):
            retrieved_docs = []
        
        # 建立對話上下文
        conversation_context = ""
        
        # 如果有Unity記憶上下文，優先使用
        if unity_memory_context:
            conversation_context = f"Unity記憶上下文:\n{unity_memory_context}\n\n"
        
        # 添加當前會話的對話歷史
        if conversation_history:
            conversation_context += "之前的對話:\n" + "\n".join([
                f"{exchange['speaker']}: {exchange['content']}" 
                for exchange in conversation_history[-5:]
            ])
        
        return conversation_context, retrieved_docs

    async def generate_persona_response(
        self,
        persona_id: str,
        current_topic: str,
        conversation_context: str,
        retrieved_docs: List[str],
        other_persona_name: str
    ) -> str:
        """生成 persona 的回應"""
        persona_data = self.chatbot.persona_loader_obj.personas[persona_id]
        
        from psychanchor.prompts import get_system_prompt
        
        system_content = get_system_prompt("persona_dialogue_response",
                                         simplified_behavior=persona_data['簡化行為'],
                                         other_persona_name=other_persona_name,
                                         memory_context="\n".join(retrieved_docs) if retrieved_docs else "（無可參考的記憶資料）",
                                         conversation_context=conversation_context)
        
        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": current_topic}
        ]
        
        return await self.chatbot.interviewer_obj.simulate_persona_answer(prompt)

    async def conduct_dialogue(
        self,
        persona1_id: str,
        persona2_id: str,
        num_rounds: int,
        initial_topic: str,
        session_id: Optional[str] = None,
        is_unity_dialogue: bool = False,
        unity_memory_context: str = ""
    ) -> Tuple[List[Dict[str, str]], str]:
        """執行兩個 persona 之間的對話"""
        await self.initialize()
        
        # 驗證 persona 是否存在
        if persona1_id not in self.chatbot.persona_loader_obj.personas:
            raise KeyError(f"Persona {persona1_id} 不存在於已載入的資料中。可用的 persona IDs: {list(self.chatbot.persona_loader_obj.personas.keys())}")
        
        if persona2_id not in self.chatbot.persona_loader_obj.personas:
            raise KeyError(f"Persona {persona2_id} 不存在於已載入的資料中。可用的 persona IDs: {list(self.chatbot.persona_loader_obj.personas.keys())}")
        
        # 獲取或創建會話 ID
        if not session_id:
            session_id = await self.create_session_id(persona1_id, persona2_id, is_unity_dialogue)
            logger.info(f"創建新的對話會話: {session_id}")
            history = []
        else:
            history = await self.load_dialogue_history(session_id)
            logger.info(f"載入會話 {session_id} 的 {len(history)} 條歷史對話")
        
        # 獲取 persona 資料
        persona1_data = self.chatbot.persona_loader_obj.personas[persona1_id]
        persona2_data = self.chatbot.persona_loader_obj.personas[persona2_id]
        
        dialogue_rounds = []
        current_topic = initial_topic
        
        for round_num in range(num_rounds):
            # Persona 1 的回合
            conv_context, retrieved_docs = await self.get_persona_context(
                persona1_id, current_topic, history, persona2_data['基本資料']['姓名'],
                unity_memory_context
            )
            
            persona1_response = await self.generate_persona_response(
                persona1_id, current_topic, conv_context, retrieved_docs,
                persona2_data['基本資料']['姓名']
            )
            
            dialogue_rounds.append({
                "speaker": persona1_data['基本資料']['姓名'],
                "content": persona1_response,
                "round": str(round_num + 1)
            })
            current_topic = persona1_response
            
            # Persona 2 的回合
            conv_context, retrieved_docs = await self.get_persona_context(
                persona2_id, current_topic, dialogue_rounds, persona1_data['基本資料']['姓名'],
                unity_memory_context
            )
            
            persona2_response = await self.generate_persona_response(
                persona2_id, current_topic, conv_context, retrieved_docs,
                persona1_data['基本資料']['姓名']
            )
            
            dialogue_rounds.append({
                "speaker": persona2_data['基本資料']['姓名'],
                "content": persona2_response,
                "round": str(round_num + 1)
            })
            current_topic = persona2_response
        
        # 保存對話歷史
        history.extend(dialogue_rounds)
        await self.save_dialogue_history(session_id, history)
        
        return dialogue_rounds, session_id 