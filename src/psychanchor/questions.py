from pathlib import Path
# QA_loader
from dataclasses import dataclass
from typing import List
import aiofiles
import asyncio

@dataclass
class Question:
    """問題類別，用於存儲單個問題的資訊"""
    id: int
    content: str

class QaLoader:
    """
    usage:    
    loader = DataLoader()
    questions = await loader.get_all_questions()
    print(f"總共有 {loader.get_question_count()} 個問題")
    
    # 顯示前三個問題作為範例
    for question in questions[:3]:
        print(f"問題 {question.id}: {question.content}")

    資料載入器，用於處理問題列表
    
    """
    
    def __init__(self, file_path: str = None):
        """
        初始化 DataLoader
        
        Args:
            file_path (str): 問題檔案的路徑
        """
        self.file_path = file_path or str(Path(__file__).with_name("questions.txt"))
        self.questions: List[Question] = []
        # 為避免race condition
        self._load_task = asyncio.create_task(self._load_questions())
    
    async def wait_until_ready(self):
        await self._load_task

    async def _load_questions(self):
        """從檔案中載入問題"""
        try:
            async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
                lines = await f.readlines()
                
            # 過濾空行並創建問題物件
            question_id = 1
            for line in lines:
                line = line.strip()
                if line and not line.startswith('```'):  # 排除空行和程式碼區塊標記
                    self.questions.append(Question(id=question_id, content=line))
                    question_id += 1
                    
        except FileNotFoundError:
            print(f"錯誤：找不到檔案 {self.file_path}")
        except Exception as e:
            print(f"載入問題時發生錯誤：{str(e)}")
    
    def get_all_questions(self) -> List[Question]:
        """獲取所有問題"""
        return self.questions
    
    def get_question_by_id(self, question_id: int) -> Question:
        """根據 ID 獲取特定問題"""
        for question in self.questions:
            if question.id == question_id:
                return question
        return None
    
    def get_question_count(self) -> int:
        """獲取問題總數"""
        return len(self.questions)

    async def add_question(self, content: str) -> Question:
        """
        添加單一問題
        
        Args:
            content (str): 問題內容
            
        Returns:
            Question: 新創建的問題物件
        """
        new_id = len(self.questions) + 1
        new_question = Question(id=new_id, content=content)
        self.questions.append(new_question)
        await self._save_questions()
        return new_question

    async def add_questions(self, contents: List[str]) -> List[Question]:
        """
        批量添加問題
        
        Args:
            contents (List[str]): 問題內容列表
            
        Returns:
            List[Question]: 新創建的問題物件列表
        """
        new_questions = []
        for content in contents:
            new_question = await self.add_question(content)
            new_questions.append(new_question)
        return new_questions

    async def _save_questions(self):
        """將問題保存到檔案"""
        try:
            async with aiofiles.open(self.file_path, 'w', encoding='utf-8') as f:
                for question in self.questions:
                    await f.write(f"{question.content}\n")
        except Exception as e:
            print(f"保存問題時發生錯誤：{str(e)}")

    async def update_question(self, question_id: int, new_content: str) -> bool:
        """
        更新指定 ID 的問題內容
        
        Args:
            question_id (int): 問題 ID
            new_content (str): 新的問題內容
            
        Returns:
            bool: 更新是否成功
        """
        for question in self.questions:
            if question.id == question_id:
                question.content = new_content
                await self._save_questions()
                return True
        return False

    async def delete_question(self, question_id: int) -> bool:
        """
        刪除指定 ID 的問題
        
        Args:
            question_id (int): 問題 ID
            
        Returns:
            bool: 刪除是否成功
        """
        for i, question in enumerate(self.questions):
            if question.id == question_id:
                self.questions.pop(i)
                # 重新編號
                for j in range(i, len(self.questions)):
                    self.questions[j].id = j + 1
                await self._save_questions()
                return True
        return False

