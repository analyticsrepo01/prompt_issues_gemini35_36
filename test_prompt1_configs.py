import os
import time
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    ThinkingConfig,
    ThinkingLevel,
)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project-0004-346516")
client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")

system_instruction = """
<instructions>
  Analyze the combined links and summaries provided within the <text> tags and provide an overall summary regarding the entity specified within the <entity> tags.

  # Output Format Constraints
Provide your output strictly in the format below.

  **CRITICAL:** Do not wrap the output in markdown code blocks (such as `xml), HTML wrapper tags, or any other introductory/concluding text. Only output the three XML tags below.

  <think>
Briefly describe the key aspects identified for overall summarization in bullet points.
</think>
<adv>ADVERSE</adv>
<sum>
Provide a brief overall summary related to the entity, strictly in English.
</sum>

  </instructions>
"""

entity_data = "<entity>Sample Entity</entity>"
text_data = """
<text>
- Link: http://news.example.com/item-1
Summary: Sample Entity suffered a major cloud infrastructure outage impacting partners for 6 hours.
- Link: http://news.example.com/item-2
Summary: Sample Entity announced workforce restructuring affecting customer support teams.
</text>
"""
full_prompt = f"{entity_data}\n{text_data}"

def run_test_set(name, thinking_cfg):
    print(f"\n--- Testing Config: {name} ---")
    config = GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2,
        thinking_config=thinking_cfg
    )
    no_resp_count = 0
    for i in range(1, 11):
        try:
            resp = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=full_prompt,
                config=config
            )
            if resp.text and resp.text.strip():
                print(f"Round {i:2d}: SUCCESS ({len(resp.text)} chars)")
            else:
                no_resp_count += 1
                cand = resp.candidates[0] if resp.candidates else None
                print(f"Round {i:2d}: NO_RESPONSE (FinishReason={cand.finish_reason if cand else 'None'})")
        except Exception as e:
            no_resp_count += 1
            print(f"Round {i:2d}: ERROR ({e})")
        time.sleep(0.5)
    print(f"Result for {name}: {10 - no_resp_count}/10 SUCCESS | {no_resp_count}/10 NO_RESPONSE")

# 1. Test ThinkingLevel.LOW (User's current Prompt 1 config)
run_test_set("ThinkingLevel.LOW", ThinkingConfig(thinking_level=ThinkingLevel.LOW))

# 2. Test ThinkingLevel.HIGH
run_test_set("ThinkingLevel.HIGH", ThinkingConfig(thinking_level=ThinkingLevel.HIGH))

# 3. Test Without ThinkingConfig
run_test_set("No ThinkingConfig", None)

