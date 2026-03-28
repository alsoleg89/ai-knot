"""agentmemo + OpenAI integration example.

This example shows how to enrich OpenAI messages with agent memory.
NOTE: Requires OPENAI_API_KEY environment variable for the actual API call.
The memory injection works without an API key.
"""

import os
import json

from agentmemo import KnowledgeBase, MemoryType
from agentmemo.integrations.openai import MemoryEnabledOpenAI

# Create a knowledge base with some facts.
kb = KnowledgeBase(agent_id="assistant")
kb.add("User prefers Python for all backend code", type=MemoryType.PROCEDURAL, importance=0.9)
kb.add("User deploys on Kubernetes with Docker", importance=0.85)
kb.add("User works at Sber managing 12 engineers", importance=0.90)
kb.add("User wants all code tested with pytest", type=MemoryType.PROCEDURAL, importance=0.85)

# Create memory-enabled wrapper.
client = MemoryEnabledOpenAI(knowledge_base=kb)

# Simulate a user message.
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write me a deployment script for my API"},
]

# Enrich messages with memory context.
enriched = client._enrich_messages(messages)

print("=== Original system prompt ===")
print(messages[0]["content"])
print()
print("=== Enriched system prompt (with memory) ===")
print(enriched[0]["content"])
print()
print("=== Full enriched messages ===")
print(json.dumps(enriched, indent=2))

# To use with the actual OpenAI API:
# import openai
# api_client = openai.OpenAI()
# response = api_client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=enriched,
# )

# Clean up.
import shutil
shutil.rmtree(".agentmemo", ignore_errors=True)
print("\nDemo complete.")
