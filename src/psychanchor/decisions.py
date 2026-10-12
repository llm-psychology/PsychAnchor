"""Extracted persona-conditioned fire decisions, not a calibrated anchoring algorithm.

The environment supplies observations; this module owns memory and model prompts.
"""
import json
import random
import logging
from typing import Dict
logger = logging.getLogger(__name__)

async def decide_action(agent, situation_context, *, rooms, connections, agents, interviewer) -> Dict:
        """使用 LLM 進行代理人決策"""
        try:
            # 建立決策 prompt
            current_room = rooms[agent.location]
            other_agents_here = [a for a in agents if a.location == agent.location and a.id != agent.id]
            available_exits = connections[agent.location]

            # 獲取 RAG 記憶
            try:
                index, docs, _ = await interviewer.load_rag_database(agent.id)
                retrieved_docs, _ = await interviewer.retrieve_similar_docs(
                    situation_context, index, docs, top_k=2
                )
            except:
                retrieved_docs = []

            # 內部工具：簡單字串裁切
            def _trim_text(text: str, max_len: int) -> str:
                if not text:
                    return ""
                text = str(text).replace('\n', ' ').strip()
                return (text[:max_len] + '…') if len(text) > max_len else text

            # 從檢索結果擷取記憶片段（10~20字，這裡選擇最長20字）
            memory_snippet = ""
            if retrieved_docs:
                try:
                    # 嘗試抓第一筆文件的主要內容
                    first_doc = retrieved_docs[0]
                    # 常見物件：有 page_content 屬性 或 字典 或 一般字串
                    if hasattr(first_doc, 'page_content'):
                        memory_snippet = _trim_text(getattr(first_doc, 'page_content', ''), 20)
                    elif isinstance(first_doc, dict):
                        memory_snippet = _trim_text(first_doc.get('page_content') or first_doc.get('content') or first_doc.get('text') or "", 20)
                    else:
                        memory_snippet = _trim_text(str(first_doc), 20)
                except Exception:
                    memory_snippet = _trim_text(str(retrieved_docs[0]), 20)

            context = f"""
當前情境: {situation_context}
你的位置: {agent.location} ({current_room})
可前往的房間: {', '.join(available_exits)}
同房間的人: {[a.name for a in other_agents_here]}
你的恐慌程度: {agent.panic_level}/100
你的健康狀況: {agent.health}/100
是否知道火災: {agent.known_fire}
最近收到的訊息: {agent.messages_received[-3:] if agent.messages_received else '無'}
            """

            from psychanchor.prompts import get_system_prompt, get_user_prompt

            # 格式化 retrieved_docs 為字符串
            retrieved_docs_str = '無特別記憶'
            if retrieved_docs:
                docs_list = retrieved_docs[:2]
                if docs_list:
                    retrieved_docs_str = '\n'.join([str(doc) for doc in docs_list])

            # 格式化 recent_messages 為字符串
            recent_messages_str = '無'
            if agent.messages_received:
                recent_msgs = agent.messages_received[-3:]
                recent_messages_str = ', '.join([msg.get('message', str(msg)) if isinstance(msg, dict) else str(msg) for msg in recent_msgs])

            # 風險評估
            room_temp = getattr(current_room, 'temperature', 25)
            room_vis = getattr(current_room, 'visibility', 100)
            room_on_fire = getattr(current_room, 'on_fire', False)
            room_smoke = getattr(current_room, 'smoke', False)
            risk_score = 0
            if room_on_fire: risk_score += 2
            if room_temp > 150: risk_score += 2
            if room_vis < 20: risk_score += 2
            if agent.panic_level >= 80: risk_score += 1
            if agent.health <= 20: risk_score += 2
            risk_level = 'low'
            if risk_score >= 6:
                risk_level = 'critical'
            elif risk_score >= 3:
                risk_level = 'high'
            elif risk_score >= 1:
                risk_level = 'medium'

            system_content = get_system_prompt("agent_decision_making",
                                             agent_name=agent.name,
                                             traits=str(agent.traits) if isinstance(agent.traits, (list, dict)) else agent.traits,
                                             social_abilities=str(agent.social_abilities) if isinstance(agent.social_abilities, (list, dict)) else agent.social_abilities,
                                             ability_attributes=str(agent.ability_attributes) if isinstance(agent.ability_attributes, (list, dict)) else agent.ability_attributes,
                                             retrieved_docs=retrieved_docs_str)

            user_content = get_user_prompt("agent_decision_making",
                                         situation_context=situation_context,
                                         location=agent.location,
                                         current_room=str(current_room),
                                         available_exits=', '.join(available_exits),
                                         other_agents=', '.join([a.name for a in other_agents_here]) if other_agents_here else '無',
                                         panic_level=agent.panic_level,
                                         health=agent.health,
                                         known_fire=agent.known_fire,
                                         recent_messages=recent_messages_str,
                                         risk_level=risk_level,
                                         room_temperature=room_temp,
                                         visibility=room_vis,
                                         on_fire=room_on_fire,
                                         smoke=room_smoke)

            prompt = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

            response = await interviewer.simulate_persona_answer(prompt)

            # 解析回應
            try:
                # 清理回應，提取JSON
                response = response.strip()
                if response.startswith('```'):
                    response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:].strip()

                decision = json.loads(response)
                # 產生精簡 10 字內獨白（根據行為與環境）
                def _generate_short_monologue(a, d: Dict) -> str:
                    # 多樣化短語，依狀態／行為與隨機性
                    import random
                    if a.escaped:
                        # 已逃生仍可返回救援
                        return "已逃生"
                    if a.injured:
                        return "受傷" if a.health > 0 else "倒地"
                    action = d.get('action', '')
                    room_state = rooms.get(a.location, None)
                    local_hazard = False
                    if room_state:
                        # 只用本地可見危險（著火/煙霧/能見度低/溫度高）作為危險依據
                        local_hazard = (
                            getattr(room_state, 'on_fire', False) or
                            getattr(room_state, 'smoke', False) or
                            getattr(room_state, 'visibility', 100) < 20 or
                            getattr(room_state, 'temperature', 25) > 150
                        )
                    # 在危險時，加入隨機多樣化
                    if local_hazard:
                        if action == 'help_others':
                            return random.choice(["我要救人", "先救人"])
                        if action == 'shout':
                            return random.choice(["呼救", "求援"])
                        if action == 'search':
                            return random.choice(["找出口", "搜尋出口"])
                        if action == 'move':
                            return random.choice(["快逃", "逃離火場"])
                        if action == 'stay':
                            return random.choice(["危險", "等一下"])
                        # 未知行為 fallback
                        return random.choice(["危險", "快逃", "呼救"]) 
                    # 沒有本地危險，但如果僅僅是收到警告（known_fire）則保持提醒語氣
                    if getattr(a, 'known_fire', False):
                        if action == 'shout':
                            return "警告火災"
                        if action == 'move':
                            return "小心前進"
                        return random.choice(["注意", "小心"]) 
                    # 非危險情境
                    if action == 'search':
                        return "找出口"
                    if action == 'help_others':
                        return random.choice(["我要救人", "先救人"])
                    if action == 'shout':
                        return "呼救"
                    if action == 'move':
                        return "移動"
                    if action == 'stay':
                        return "目前安全"
                    return "等待"
                agent.internal_monologue = _trim_text(_generate_short_monologue(agent, decision), 10)
                # 保留原本記憶片段欄位，即使前端不顯示
                agent.memory_snippet = memory_snippet
                return decision
            except json.JSONDecodeError:
                # 如果解析失敗，返回預設行為
                logger.warning(f"{agent.name} 決策解析失敗，使用預設行為")
                # 簡化失敗提示
                agent.internal_monologue = _trim_text("等待", 10)
                agent.memory_snippet = memory_snippet
                return {
                    "action": "move" if agent.known_fire else "stay",
                    "target": random.choice(available_exits) if agent.known_fire else agent.location,
                    "message": "",
                    "reasoning": "解析失敗的預設行為"
                }

        except Exception as e:
            logger.error(f"代理人 {agent.name} 決策失敗: {e}")
            # 決策錯誤時也維持最新的記憶片段供可視化
            try:
                agent.memory_snippet = memory_snippet if 'memory_snippet' in locals() else ""
            except Exception:
                pass
            agent.internal_monologue = _trim_text("錯誤", 10)
            return {
                "action": "stay",
                "target": agent.location,
                "message": "",
                "reasoning": "決策系統錯誤"
            }
