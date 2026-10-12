from personadough.storage import persona_directory
# PERSONA_loader.py
import os
import json
from typing import List, Dict, Optional
import shutil
from datetime import datetime
import aiofiles
import asyncio
import time

class PersonaLoader:
    """usage:   personas = PersonaLoader()
    
                await personas.wait_until_ready() #important: or it will return null
    """
    def __init__(self, database_dir=None):
        self.database_dir = str(persona_directory(database_dir))
        self.backup_dir = os.path.join(self.database_dir, "backup")
        self.personas = {}  # 儲存所有載入的角色資料
        # 為避免race condition
        
        self._load_task = asyncio.create_task(self._load_all_personas())
    
    async def wait_until_ready(self):
        await self._load_task

    async def _load_all_personas(self):
        """載入資料庫目錄下的所有角色資料"""
        if not os.path.exists(self.database_dir):
            os.makedirs(self.database_dir, exist_ok=True)
            return

        for filename in os.listdir(self.database_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.database_dir, filename)
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        persona_data = json.loads(content)
                        persona_id = persona_data["基本資料"]["id"]
                        self.personas[persona_id] = persona_data
                except Exception as e:
                    print(f"載入檔案 {filename} 時發生錯誤：{str(e)}")

    def get_all_personas(self) -> Dict[str, Dict]:
        """獲取所有角色資料"""
        return self.personas

    def get_persona_by_id(self, persona_id: str) -> Optional[Dict]:
        """根據ID獲取特定角色資料"""
        return self.personas.get(persona_id)

    def get_persona_by_name(self, name: str) -> List[Dict]:
        """根據姓名獲取角色資料（可能有多個同名角色）"""
        return [
            persona for persona in self.personas.values()
            if persona["基本資料"]["姓名"] == name
        ]

    def get_persona_count(self) -> int:
        """獲取角色總數"""
        return len(self.personas)

    def get_persona_basic_list(self) -> List[Dict]:
        """獲取所有角色的基本資料列表"""
        return [
            {
                "id": persona["基本資料"]["id"],
                "姓名": persona["基本資料"]["姓名"],
                "性別": persona["基本資料"]["性別"],
                "年紀": persona["基本資料"]["年紀"]
            }
            for persona in self.personas.values()
        ]
    
    async def add_persona(self, persona_data: Dict) -> bool:
        """
        添加新角色
        
        Args:
            persona_data: 角色資料字典
            
        Returns:
            bool: 是否成功添加
        """
        try:
            persona_id = persona_data["基本資料"]["id"]
            name = persona_data["基本資料"]["姓名"]
            filename = f"{persona_id}_{name}.json"
            file_path = os.path.join(self.database_dir, filename)
            
            # 檢查是否已存在
            if persona_id in self.personas:
                print(f"角色 ID {persona_id} 已存在")
                return False
                
            # 保存檔案
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(persona_data, ensure_ascii=False, indent=2))
            
            # 更新記憶體中的資料
            self.personas[persona_id] = persona_data
            return True
            
        except Exception as e:
            print(f"添加角色時發生錯誤：{str(e)}")
            return False

    async def update_persona(self, persona_id: str, new_data: Dict) -> bool:
        """
        更新角色資料
        
        Args:
            persona_id: 角色ID
            new_data: 新的角色資料
            
        Returns:
            bool: 是否成功更新
        """
        try:
            if persona_id not in self.personas:
                print(f"找不到角色 ID {persona_id}")
                return False
                
            # 備份原始檔案
            await self._backup_persona(persona_id)
            
            # 更新檔案
            old_data = self.personas[persona_id]
            name = old_data["基本資料"]["姓名"]
            filename = f"{persona_id}_{name}.json"
            file_path = os.path.join(self.database_dir, filename)
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(new_data, ensure_ascii=False, indent=2))
            
            # 更新記憶體中的資料
            self.personas[persona_id] = new_data
            return True
            
        except Exception as e:
            print(f"更新角色時發生錯誤：{str(e)}")
            return False

    async def delete_persona(self, persona_id: str) -> bool:
        """
        刪除角色
        
        Args:
            persona_id: 角色ID
            
        Returns:
            bool: 是否成功刪除
        """
        try:
            if persona_id not in self.personas:
                print(f"找不到角色 ID {persona_id}")
                return False
                
            # 備份原始檔案
            await self._backup_persona(persona_id)
            
            # 刪除檔案
            persona_data = self.personas[persona_id]
            name = persona_data["基本資料"]["姓名"]
            filename = f"{persona_id}_{name}.json"
            file_path = os.path.join(self.database_dir, filename)
            
            os.remove(file_path)
            
            # 更新記憶體中的資料
            del self.personas[persona_id]
            return True
            
        except Exception as e:
            print(f"刪除角色時發生錯誤：{str(e)}")
            return False

    async def _backup_persona(self, persona_id: str):
        """備份角色資料"""
        try:
            if not os.path.exists(self.backup_dir):
                os.makedirs(self.backup_dir, exist_ok=True)
                
            persona_data = self.personas[persona_id]
            name = persona_data["基本資料"]["姓名"]
            filename = f"{persona_id}_{name}.json"
            source_path = os.path.join(self.database_dir, filename)
            
            # 使用時間戳記作為備份檔案名稱
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{persona_id}_{name}_{timestamp}.json"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            # 複製檔案
            shutil.copy2(source_path, backup_path)
            
        except Exception as e:
            print(f"備份角色資料時發生錯誤：{str(e)}")

    def search_personas(self, keyword: str) -> List[Dict]:
        """
        搜尋角色（根據姓名、ID或任何包含關鍵字的欄位）
        
        Args:
            keyword: 搜尋關鍵字
            
        Returns:
            List[Dict]: 符合條件的角色列表
        """
        results = []
        keyword = keyword.lower()
        
        for persona in self.personas.values():
            # 將所有值轉換為字串並搜尋
            persona_str = json.dumps(persona, ensure_ascii=False).lower()
            if keyword in persona_str:
                results.append(persona)
                
        return results

    def get_persona_statistics(self) -> Dict:
        """
        獲取角色資料統計資訊
        
        Returns:
            Dict: 包含各種統計資訊的字典
        """
        stats = {
            "總角色數": len(self.personas),
            "性別統計": {},
            "年齡分布": {
                "18-25": 0,
                "26-35": 0,
                "36-45": 0,
                "46-55": 0,
                "56-65": 0
            }
        }
        
        for persona in self.personas.values():
            # 性別統計
            gender = persona["基本資料"]["性別"]
            stats["性別統計"][gender] = stats["性別統計"].get(gender, 0) + 1
            
            # 年齡分布
            age = persona["基本資料"]["年紀"]
            if 18 <= age <= 25:
                stats["年齡分布"]["18-25"] += 1
            elif 26 <= age <= 35:
                stats["年齡分布"]["26-35"] += 1
            elif 36 <= age <= 45:
                stats["年齡分布"]["36-45"] += 1
            elif 46 <= age <= 55:
                stats["年齡分布"]["46-55"] += 1
            elif 56 <= age <= 65:
                stats["年齡分布"]["56-65"] += 1
                
        return stats

    async def reload(self):
        """重新載入所有角色資料"""
        self.personas.clear()
        await self._load_all_personas()

# =================================================================

