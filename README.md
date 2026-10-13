# PsychAnchor

接收 PersonaDough 產生的人格 JSON，探索回答與行為的一致性：人格是傾向而不是禁令，外向角色也可能獨處，不能把單次例外直接判為人格失效。

## 目前完成到哪裡

本次由 PersonaDough 拆入：人格資料載入、標準與動態訪談、問答 embedding/FAISS 檢索、單人/雙人/群組對話，以及既有火災情境的人格決策提示與解析。

**目前只有人格提示與記憶檢索。尚未實作量化人格偏離、跨情境一致性評估或自動校正的錨定演算法。** 原始研究構想保留在 [這邊](https://github.com/llm-psychology/PsychAnchor/blob/refactor/extract-anchoring/docs/original-design.md)。

## 邊界

- 讀取人格 JSON；不重新生成或改寫初始人格。
- 訪談問答、對話與後續經歷屬於此 repo 的資料。
- 產生回答與情境決策；環境狀態、規則執行、HTTP 與 SSE 由 API/模擬 adapter 負責。
- 尚未完成的後續故事生成與事件記憶流，見主 repo 的資料格式與交換規則。

## 安裝與使用

先將 PersonaDough 與此 repo 放在相鄰資料夾。需要 Python 3.10+；完整取得方式見 [Phersu 主導覽](https://github.com/llm-psychology/Phersu)。

```powershell
python -m pip install -e ../PersonaDough -e .
Copy-Item .env.example .env
python -m psychanchor --persona ../PersonaDough/examples/5c771e5f3c_李靜宜.json --question '你喜歡怎樣認識新朋友？'
```

OpenAI/Ollama 聊天模式由 `LLM_PROVIDER` 選擇。此命令直接以人格回答，不會自動建立訪談索引
使用 `--memory` 加入已建立的索引 ID。
embedding 使用上仍需 OpenAI key。

完整訪談索引流程可用 `psychanchor.interviewer.generate_persona_rag_by_id(id)`。需先讓 `PERSONA_DATA_DIR` 指向包含該角色 JSON 的資料夾。

## 資料與依賴

- `PERSONA_DATA_DIR`：唯讀輸入人格目錄（舊管理方法仍保留，不應用於錨定期間修改初始人格）。
- `PSYCHANCHOR_DATA_DIR`：預設 `data/psychanchor`，索引放在其中的 `rag/{id}/`。
- `CONVERSATION_DATA_DIR`：預設 `data/conversations`。
- `personadough`：共用 JSON 格式驗證、資料路徑與模型 adapter；不呼叫生成器。
`decisions.py` 保留原火災決策邏輯，接收呼叫端提供的房間與角色觀測，沒有引入 API package。它是情境策略原型，不代表通用心理錨定。

## 驗證與限制

```powershell
python -m pip install -e '../PersonaDough[dev]' -e '.[dev]'
python -m pytest tests
```

本專案採用 GPL-3.0 授權，詳見。
