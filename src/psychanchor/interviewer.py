from psychanchor.storage import rag_directory, history_directory
# interviewer_generator.py
import faiss
from typing import List, Dict
import numpy as np
import os
import json
import asyncio
import aiofiles
import time

from personadough.runtime.responder import LLM_responder
from psychanchor.questions import QaLoader
from psychanchor.persona_loader import PersonaLoader


class Interviewer(LLM_responder):
    def __init__(self):
        super().__init__()
        self.rag_data_dir = str(rag_directory())
        os.makedirs(self.rag_data_dir, exist_ok=True)
        self.batch_size = 10  # 每批次的問題數量

    # ========== 1. Collect user answers ==========
    # 只有在humanoid_database是空的時候才會用到
    async def collect_user_answers(self, question_list: List[str]) -> List[Dict[str, str]]:
        qa_pairs = []
        for q in question_list:
            a = input(f"{q}\n> ")
            qa_pairs.append({"q": q, "a": a})
        return qa_pairs

    # ========== 2. Format QA Pairs ==========
    def format_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> List[str]:
        return [f"Q：{pair['q']} A：{pair['a']}" for pair in qa_pairs]

    # ========== 3. Generate Embeddings ==========
    # 使用父類別 LLM_responder 的 generate_embedding 方法

    # ========== 4. Build Vector Index ==========
    def build_vector_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    # ========== 5. Retrieve Similar Docs ==========
    async def retrieve_similar_docs(self, query: str, index: faiss.IndexFlatL2, docs: List[str], top_k: int = 2, similarity_threshold: float = 0.7) -> tuple:
        if top_k < 1 or index.ntotal == 0 or not docs:
            return [], np.array([], dtype=np.float32)
        respond = await self.get_embedding_for_text(query)
        query_vector = np.array([respond.data[0].embedding])
        D, I = index.search(query_vector.astype(np.float32), min(top_k, index.ntotal))
        selected = [(docs[i], float(d)) for i, d in zip(I[0], D[0])
                    if 0 <= i < len(docs) and (similarity_threshold is None or d <= similarity_threshold)]
        return [doc for doc, _ in selected], np.array([d for _, d in selected])

    # ========== 6. Build Simulation Prompt ==========
    def build_simulation_prompt(self, persona, retrieved_docs: List[str], user_query: str) -> List[Dict[str, str]]:
        from psychanchor.prompts import get_system_prompt
        
        memory_context = "\n".join(retrieved_docs) if retrieved_docs else "（無可參考的記憶資料）"
        system_prompt = get_system_prompt("build_simulation_prompt", 
                                        simplified_behavior=persona['簡化行為'],
                                        memory_context=memory_context)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

    # ========== 7. Simulate Persona Answer ==========
    # 使用父類別 LLM_responder 的 simulate_persona_answer 方法

    # ========== 8. Save RAG Database ==========
    async def save_rag_database(self, name: str, embeddings: np.ndarray, index: faiss.IndexFlatL2, docs: List[str], qa_pairs: List[Dict[str, str]]):
        """
        儲存 RAG 資料庫
        
        Args:
            name (str): 資料庫名稱
            embeddings (np.ndarray): 文件嵌入向量（用於建立 index）
            index (faiss.IndexFlatL2): FAISS 索引
            docs (List[str]): 格式化後的文件（用於建立 embeddings）
            qa_pairs (List[Dict[str, str]]): 原始問答對
        """
        db_dir = os.path.join(self.rag_data_dir, name)
        os.makedirs(db_dir, exist_ok=True)
        
        # 儲存 FAISS index（包含所有向量資訊）
        async with aiofiles.open(os.path.join(db_dir, "index.faiss"), "wb") as f:
            await f.write(faiss.serialize_index(index).tobytes())
        
        # 儲存原始問答對
        async with aiofiles.open(os.path.join(db_dir, "qa_pairs.json"), "w", encoding="utf-8") as f:
            await f.write(json.dumps(qa_pairs, ensure_ascii=False, indent=2))

    # ========== 9. Load RAG Database ==========
    async def load_rag_database(self, name: str) -> tuple:
        """
        載入 RAG 資料庫
        
        Args:
            name (str): 資料庫名稱
            
        Returns:
            tuple: (index, docs, qa_pairs)
        """
        db_dir = os.path.join(self.rag_data_dir, name)
        
        # 載入 FAISS index（包含所有向量資訊）
        async with aiofiles.open(os.path.join(db_dir, "index.faiss"), "rb") as f:
            index = faiss.deserialize_index(np.frombuffer(await f.read(), dtype=np.uint8).copy())
        
        # 載入原始問答對
        async with aiofiles.open(os.path.join(db_dir, "qa_pairs.json"), "r", encoding="utf-8") as f:
            content = await f.read()
            qa_pairs = json.loads(content)
            
        # 動態生成 docs
        docs = self.format_qa_pairs(qa_pairs)
            
        return index, docs, qa_pairs

    async def process_questions_in_batches(self, qaloader: QaLoader, persona_loader: PersonaLoader = None, persona_id: str = None) -> List[Dict[str, str]]:
        """
        批次處理問題
        
        Args:
            qaloader (DataLoader): 問題載入器
            persona_loader (PersonaLoader, optional): 角色載入器
            persona_id (str, optional): 要使用的角色ID
            
        Returns:
            List[Dict[str, str]]: 所有問答對
        """
        all_qa_pairs = []
        questions = qaloader.get_all_questions()
        total_questions = len(questions)
        
        # 如果提供了角色ID，獲取角色資料
        persona_data = None
        if persona_loader and persona_id:
            persona_data = persona_loader.get_persona_by_id(persona_id)
            if not persona_data:
                print(f"找不到ID為 {persona_id} 的角色")
                return all_qa_pairs
            print(f"\n使用角色：{persona_data['基本資料']['姓名']}")
        
        for i in range(0, total_questions, self.batch_size):
            batch_questions = questions[i:i + self.batch_size]
            print(f"\n=== 批次 {i//self.batch_size + 1}/{(total_questions-1)//self.batch_size + 1} ===")
            print(f"處理問題 {i+1} 到 {min(i+self.batch_size, total_questions)}")
            
            # 取得當前批次的問題內容
            batch_question_contents = [q.content for q in batch_questions]
            batch_qa_pairs = []  # 初始化批次問答對列表
            
            # 如果有角色資料，使用角色自動回答
            if persona_data:
                from psychanchor.prompts import get_user_prompt
                
                # 將當前批次的問題加入 prompt
                questions_text = "\n".join(batch_question_contents)
                question_batch_prompt = get_user_prompt("batch_interview_questions", questions=questions_text)
                
                print("正在生成回答...")

                sys_prompt = f"你現在是\n {json.dumps(persona_data, ensure_ascii=False)}"
                
                answer = await self.full_chat_gpt_41(sys_prompt, question_batch_prompt, 0.7)

                try:
                    # 解析 JSON 回答
                    # 移除可能的開頭和結尾空白
                    answer = answer.strip().replace("【","").replace("【輸出】","").replace("】","").replace("[輸出]","").replace("輸出：", "")
                    
                    # 解析 JSON
                    qa_list = json.loads(answer.removeprefix("```json").removeprefix("```").removesuffix("```").strip(), strict=False)
                    
                    # 將解析後的問答對加入批次結果
                    for qa in qa_list:
                        batch_qa_pairs.append({
                            "q": qa["問題"],
                            "a": qa["回答"]
                        })
                        print(f"\n問題：{qa['問題']}")
                        print(f"回答：{qa['回答']}")
                        
                except json.JSONDecodeError as e:
                    print(f"解析回答時發生錯誤：{str(e)}")
                    print("<原始回答>", answer)
                    # 如果解析失敗，使用原始方式處理
                    for question in batch_question_contents:
                        batch_qa_pairs.append({
                            "q": question,
                            "a": "解析回答失敗"
                        })
            else:
                # 如果沒有角色資料，則收集使用者回答
                print("沒有角色資料，將收集你的回答")
                batch_qa_pairs = await self.collect_user_answers(batch_question_contents)
            
            all_qa_pairs.extend(batch_qa_pairs)
            print(f"完成批次 {i//self.batch_size + 1}")
            
        return all_qa_pairs

    async def process_dynamic_interview(self, persona_id: str, persona_data: dict, static_interview_data:str, num_rounds: int = 10) -> List[Dict[str, str]]:
        """
        進行動態訪談，根據回答產生後續問題
        
        Args:
            persona_id (str): 角色ID
            persona_data (dict): 角色資料
            num_rounds (int): 訪談回合數
            
        Returns:
            List[Dict[str, str]]: 問答對列表
        """
        # ==============================
        qa_pairs = []
        current_context = ""
        # ==============================
        from psychanchor.prompts import get_system_prompt, get_user_prompt, get_prompt
        
        sys_prompt = get_system_prompt("dynamic_interview_generator")
        prompt_data = get_prompt("dynamic_interview_generator", template_keys=["system", "context_template"], static_interview_data=static_interview_data)
        question_batch_prompt = prompt_data["context_template"]
        # ==============================
        # 初始問題
        initial_question = await self.full_chat_gpt_41_mini(sys_prompt, question_batch_prompt, 0.7) # 5000 tokens
        qa_pairs.append({"q": initial_question, "a": ""})
        # ==============================
        for round_num in range(num_rounds):
            print(f"\n=== {persona_id}的訪談回合 {round_num + 1}/{num_rounds} ===")
            
            # 取得當前問題
            current_question = qa_pairs[-1]["q"]
            print(f"\n問題：{current_question}")

            sys_prompt = f"你現在是\n {json.dumps(persona_data, ensure_ascii=False)}"

            # 使用角色回答問題
            prompt_data = get_prompt("dynamic_interview_generator", template_keys=["single_response_template"], current_question=current_question)
            question_batch_prompt = prompt_data["single_response_template"]
            
            
            answer = await self.full_chat_gpt_41(sys_prompt, question_batch_prompt, 0.7)
            
            try:
                # 解析回答
                answer = answer.strip().replace("【","").replace("【輸出】","").replace("】","").replace("[輸出]","").replace("輸出：", "")
                qa_list = json.loads(answer.removeprefix("```json").removeprefix("```").removesuffix("```").strip(), strict=False)
                current_answer = qa_list[0]["回答"]
                qa_pairs[-1]["a"] = current_answer
                print(f"回答：{current_answer}")
                
                # 更新上下文
                current_context += f"Q：{current_question}\nA：{current_answer}\n\n"
                
                # 使用 MINI_MODEL 產生下一個問題
                next_q_prompts = get_prompt("dynamic_interview_generator", template_keys=["next_question_system", "next_question_template"], current_context=current_context)
                sys_prompt = next_q_prompts["next_question_system"]
                ask_prompt = next_q_prompts["next_question_template"]
                
                next_question = await self.full_chat_gpt_41_mini(sys_prompt, ask_prompt, 0.9)
                next_question = next_question.strip()
                
                # 如果不是最後一回合，加入新問題
                if round_num < num_rounds - 1:
                    qa_pairs.append({"q": next_question, "a": ""})
                
            except json.JSONDecodeError as e:
                print(f"解析回答時發生錯誤：{str(e)}")
                print("<原始回答>", answer)
                break
                
        return qa_pairs

# ========== Main Example Flow ==========
async def process_persona(persona, interviewer:Interviewer, qaloader:QaLoader, persona_loader:PersonaLoader, rag_data_dir, existing_dbs):
    # bug: persona只有基本資料
    # persona 是一整個一個人的json
    persona_id = persona['基本資料']['id']
    
    if persona_id in existing_dbs and all(os.path.isfile(os.path.join(rag_data_dir, persona_id, filename)) for filename in ("index.faiss", "qa_pairs.json")):
        print(f"\n跳過角色 {persona_id}（資料庫已存在）")
        return
    
    print(f"\n處理角色：{persona['基本資料']['姓名']} (ID: {persona_id})")
    print(f"總共有 {qaloader.get_question_count()} 個問題")
    print(f"將以每 {interviewer.batch_size} 個問題為一批次進行")

    qa_pairs = await interviewer.process_questions_in_batches(qaloader, persona_loader, persona_id)
    

    print("\n=== 開始動態訪談 ===")
    qa_pairs += await interviewer.process_dynamic_interview(persona_id, persona, str(qa_pairs), num_rounds=10)

    print("\n=== 建立記憶庫 ===")
    docs = interviewer.format_qa_pairs(qa_pairs)
    embeddings = await interviewer.generate_embedding(docs)
    index = interviewer.build_vector_index(embeddings)

    await interviewer.save_rag_database(persona_id, embeddings, index, docs, qa_pairs)
    print(f"資料庫已自動儲存為：{persona_id}")

# =========================================================
async def generate_persona_rag_by_id(id:str):
    # this unit test can generate all the interview result in humanoid database
    interviewer = Interviewer()
    qaloader = QaLoader()
    await qaloader.wait_until_ready() #important: or it will return null
    persona_loader = PersonaLoader()
    await persona_loader.wait_until_ready() #important: or it will return null

    persona_list = persona_loader.get_all_personas() #抓取所有persona
    rag_data_dir = str(rag_directory())
    os.makedirs(rag_data_dir, exist_ok=True)
    
    existing_dbs = set()
    if os.path.exists(rag_data_dir):
        existing_dbs = set(d for d in os.listdir(rag_data_dir) 
                          if os.path.isdir(os.path.join(rag_data_dir, d)))
    print(f"幫 {id} 生成RAG資料庫")
    await process_persona(persona_list[id], interviewer, qaloader, persona_loader, rag_data_dir, existing_dbs)

async def unit_test():
    # this unit test can generate all the interview result in humanoid database
    interviewer = Interviewer()
    qaloader = QaLoader()
    await qaloader.wait_until_ready() #important: or it will return null
    persona_loader = PersonaLoader()
    await persona_loader.wait_until_ready() #important: or it will return null

    persona_list = persona_loader.get_all_personas() #抓取所有persona
    rag_data_dir = str(rag_directory())
    os.makedirs(rag_data_dir, exist_ok=True)
    
    existing_dbs = set()
    if os.path.exists(rag_data_dir):
        existing_dbs = set(d for d in os.listdir(rag_data_dir) 
                          if os.path.isdir(os.path.join(rag_data_dir, d)))
    
    print("\n=== 開始處理角色資料庫 ===")
    print(f"總共有 {len(persona_list)} 個角色")

    # 建立最大一組2個 coroutine 任務
    sem = asyncio.Semaphore(2)

    async def limited_process_persona(persona):
        async with sem:
            await process_persona(persona, interviewer, qaloader, persona_loader, rag_data_dir, existing_dbs)

    await asyncio.gather(*(limited_process_persona(persona_list[p]) for p in persona_list)) #p是id list
    print("\n=== 所有角色處理完成 ===")

if __name__ == "__main__":
    start = time.time()
    asyncio.run(unit_test())# 8個persona花費650秒
    end = time.time()
    print(f"\n✅ 全部任務完成，共花費 {end - start:.2f} 秒")


