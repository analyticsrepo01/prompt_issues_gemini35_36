import os
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    ThinkingConfig,
    ThinkingLevel,
)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "my-project-0004-346516")

client = genai.Client(
    vertexai=True, project=PROJECT_ID, location="asia-northeast1"
)

system_instruction = """<instructions>
  I am a sourcing manager seeking to keep track of news regarding key suppliers for my bank. My objective is to summarize all negative developments that may indicate a potential business disruption, degradation, or change in the products or services provided by the supplier.

  Please analyze the combined links and summaries provided within the <text> tags and provide an overall summary of the relevant adverse developments regarding the specified supplier within the <entity> tags.

  # Output Format Constraints
Provide your output strictly in the format below.

  **CRITICAL:** Do not wrap the output in markdown code blocks (such as `xml), HTML wrapper tags, or any other introductory/concluding text. Only output the three XML tags below.

  <think>
Briefly describe the adverse aspects identified for overall summarization in bullet points.
</think>
<adv>ADVERSE</adv>
<sum>
Provide a brief overall summary of the adverse developments related to the entity, strictly in English.
</sum>

  </instructions>"""

entity_data = "<entity>Acme Corp</entity>"
text_data = """<text>
- Link: http://news.example.com/acme-outage
Summary: Acme Corp suffered a major cloud infrastructure outage impacting financial partners for 6 hours.
- Link: http://news.example.com/acme-layoffs
Summary: Acme Corp announced a 15% reduction in force affecting their customer support teams.
</text>"""

full_prompt = f"{entity_data}\n{text_data}"

print("=== Test 1: ThinkingConfig(thinking_level=ThinkingLevel.LOW) ===")
config1 = GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.2,
    thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.LOW),
)

try:
    response1 = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=full_prompt,
        config=config1,
    )
    print("response.text:", repr(response1.text))
    if response1.candidates:
        c = response1.candidates[0]
        print("finish_reason:", c.finish_reason)
        print("content:", c.content)
        if c.content and c.content.parts:
            for idx, part in enumerate(c.content.parts):
                print(f"Part {idx}: thought={getattr(part, 'thought', None)}, text={repr(getattr(part, 'text', None))}")
except Exception as e:
    print("Error Test 1:", e)

print("\n=== Test 2: ThinkingConfig(thinking_level=ThinkingLevel.LOW) ===")
config2 = GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.2,
    thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.LOW),
)

try:
    response2 = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=full_prompt,
        config=config2,
    )
    print("response.text:", repr(response2.text))
    if response2.candidates:
        c = response2.candidates[0]
        print("finish_reason:", c.finish_reason)
        print("content:", c.content)
        if c.content and c.content.parts:
            for idx, part in enumerate(c.content.parts):
                print(f"Part {idx}: thought={getattr(part, 'thought', None)}, text={repr(getattr(part, 'text', None))}")
except Exception as e:
    print("Error Test 2:", e)

print("\n=== Test 3: Without ThinkingConfig ===")
config3 = GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.2,
)

try:
    response3 = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=full_prompt,
        config=config3,
    )
    print("response.text:", repr(response3.text))
except Exception as e:
    print("Error Test 3:", e)
