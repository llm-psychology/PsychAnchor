import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from psychanchor.decisions import decide_action


def test_decision_uses_persona_and_returns_model_action():
    agent = SimpleNamespace(id="5c771e5f3c", name="李靜宜", location="客廳", traits=["外向活潑"],
        social_abilities=["善於溝通"], ability_attributes=["團隊合作"], panic_level=0,
        health=100, known_fire=False, messages_received=[], escaped=False, injured=False)
    room = SimpleNamespace(temperature=25, visibility=100, on_fire=False, smoke=False)
    interviewer = SimpleNamespace(load_rag_database=AsyncMock(side_effect=FileNotFoundError),
        simulate_persona_answer=AsyncMock(return_value='{"action":"stay","target":"客廳","message":"","reasoning":"先觀察"}'))
    result = asyncio.run(decide_action(agent,"朋友聊天",rooms={"客廳":room},connections={"客廳":["走廊"]},agents=[agent],interviewer=interviewer))
    assert result["action"] == "stay"
    prompt = interviewer.simulate_persona_answer.call_args.args[0]
    assert "外向活潑" in prompt[0]["content"]
    assert agent.memory_snippet == ""
