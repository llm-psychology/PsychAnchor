import argparse
import asyncio
import json
from pathlib import Path
from personadough.contract import validate_persona
from .interviewer import Interviewer

def main():
    parser = argparse.ArgumentParser(description="Answer as an existing persona")
    parser.add_argument("--persona", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--memory", help="Optional existing RAG index ID")
    args = parser.parse_args()
    persona = validate_persona(json.loads(Path(args.persona).read_text(encoding="utf-8")))
    async def run():
        interviewer = Interviewer()
        docs = []
        if args.memory:
            if args.memory != persona["基本資料"]["id"]:
                parser.error("--memory must match the persona ID")
            index, documents, _ = await interviewer.load_rag_database(args.memory)
            docs, _ = await interviewer.retrieve_similar_docs(args.question, index, documents)
        prompt = interviewer.build_simulation_prompt(persona, docs, args.question)
        print(await interviewer.simulate_persona_answer(prompt))
    asyncio.run(run())
