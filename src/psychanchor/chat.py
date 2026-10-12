from psychanchor.storage import rag_directory, history_directory
# chat_with_persona.py
import os
import asyncio
from psychanchor.interviewer import Interviewer
from psychanchor.persona_loader import PersonaLoader
import time

class ChatWith():

    def __init__(self):
        self.interviewer_obj = Interviewer()
        self.persona_loader_obj = PersonaLoader()
        self._load_task = asyncio.create_task(self.persona_loader_obj.wait_until_ready())
    
    async def wait_until_ready(self):
        await self._load_task

    async def list_available_personas(self):
        """列出所有可用的角色rag資料庫"""
        rag_data_dir = str(rag_directory())
        if not os.path.exists(rag_data_dir):
            print("尚未建立任何角色資料庫")
            return []
        db_dirs = [d for d in os.listdir(rag_data_dir) # 獲取所有資料庫目錄
                if os.path.isdir(os.path.join(rag_data_dir, d))]
        if not db_dirs:
            print("尚未建立任何角色資料庫")
            return []
        available_personas = []
        print("\n可用的角色資料庫：")
        for db_id in db_dirs:
            try:
                persona_data = self.persona_loader_obj.get_persona_by_id(db_id)
                if persona_data:
                    print(f"ID: {db_id}")
                    print(f"姓名: {persona_data['基本資料']['姓名']}")
                    print(f"性別: {persona_data['基本資料']['性別']}")
                    print(f"年紀: {persona_data['基本資料']['年紀']}")
                    print("-" * 30)
                    available_personas.append(db_id)
            except Exception as e:
                print(f"載入角色 {db_id} 時發生錯誤：{str(e)}")
        return available_personas

    async def chat_with_persona(self, persona_id: str):
        """與指定角色進行對話"""
        try:
            # 載入角色的 RAG 資料庫
            index, docs, qa_pairs = await self.interviewer_obj.load_rag_database(persona_id)
            print(f"\n成功載入角色資料庫：{persona_id}")
            
            # 開始問答循環
            while True:
                query = input("\n請輸入你的提問（或輸入 'exit' 離開）：\n> ")
                if query.strip().lower() == "exit":
                    break
                    
                retrieved, distances = await self.interviewer_obj.retrieve_similar_docs(query, index, docs)
                print("\n=== 相似度分析 ===")
                print(f"查詢問題：{query}")
                print("\n最相似的資料：")
                
                for i, (doc, dist) in enumerate(zip(retrieved, distances)):
                    print(f"\n{i+1}. 距離：{dist:.4f}")
                    print(f"內容：{doc}")
                    if dist > 0.5:  # 如果距離太遠，加入提示
                        print("(注意：此資料相似度較低)")
                        retrieved = None
                    else:
                        print("(相似度高)")
                        
                prompt = self.interviewer_obj.build_simulation_prompt(self.persona_loader_obj.personas[persona_id], retrieved, query)
                answer = await self.interviewer_obj.simulate_persona_answer(prompt)
                print(f"\n模擬人格回答：\n{answer}\n")
                
        except Exception as e:
            print(f"發生錯誤：{str(e)}")

# =========================================================

async def main():
    """主程式"""
    print("=== 角色對話系統 ===")
    chatobj = ChatWith()
    await chatobj.wait_until_ready()
    # 列出所有可用的角色
    available_personas = await chatobj.list_available_personas()
    
    if not available_personas:
        return
    
    # 讓使用者選擇角色
    while True:
        persona_id = input("\n請輸入要對話的角色ID（或輸入 'exit' 離開）：")
        if persona_id.lower() == 'exit':
            break
            
        if persona_id in available_personas:
            await chatobj.chat_with_persona(persona_id)
        else:
            print(f"找不到ID為 {persona_id} 的角色資料庫")

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"\n✅ 全部任務完成，共花費 {end - start:.2f} 秒")