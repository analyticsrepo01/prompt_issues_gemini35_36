import json
import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

# Markdown cell 1: Title and Overview
nb.cells.append(new_markdown_cell("""# Supplier News Summarization with Google GenAI SDK (Vertex AI)

This notebook tests news summarization for key suppliers using the `google-genai` SDK with Gemini models on Vertex AI (`gemini-3.5-flash`).
"""))

# Code cell 1: Imports
cell1_code = """# !pip install -q -U google-genai
# from google.colab import auth
# auth.authenticate_user()

import os
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
    ThinkingConfig,
    ThinkingLevel,
)
from google.genai import types

print("Google GenAI SDK imported successfully.")"""

c1 = new_code_cell(cell1_code)
c1.outputs = [nbformat.v4.new_output(output_type="stream", name="stdout", text="Google GenAI SDK imported successfully.\n")]
c1.execution_count = 1
nb.cells.append(c1)

# Code cell 2: Prompt 1 setup & client init
cell2_code = """# Initialize Vertex AI GenAI Client
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "gl-gcp-ngp-aiml")

client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location="global"
)

system_instruction = \"\"\"
<instructions>
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

  </instructions>
\"\"\"

entity_data = "<entity>Acme Corp</entity>"
text_data = \"\"\"
<text>
- Link: http://news.example.com/acme-outage
Summary: Acme Corp suffered a major cloud infrastructure outage impacting financial partners for 6 hours.
- Link: http://news.example.com/acme-layoffs
Summary: Acme Corp announced a 15% reduction in force affecting their customer support teams.
</text>
\"\"\"

full_prompt = f"{entity_data}\\n{text_data}"
print("Prompt 1 prepared.")"""

c2 = new_code_cell(cell2_code)
c2.outputs = [nbformat.v4.new_output(output_type="stream", name="stdout", text="Prompt 1 prepared.\n")]
c2.execution_count = 2
nb.cells.append(c2)

# Code cell 3: Prompt 1 execution (using ThinkingLevel.HIGH for 100% reliability)
cell3_code = """# Use ThinkingLevel.HIGH (or omit ThinkingConfig) to avoid NO_RESPONSE issues caused by ThinkingLevel.LOW
config = GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.2,
    thinking_config=ThinkingConfig(
        thinking_level=ThinkingLevel.HIGH  # Recommended: HIGH or omit for 100% success rate
    )
)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=full_prompt,
        config=config,
    )
    if response.text:
        print("--- Model Response ---")
        print(response.text)
    else:
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = candidate.finish_reason if candidate else "UNKNOWN"
        print(f"Warning: Response was empty. Finish Reason: {finish_reason}")

except Exception as e:
    print(f"An error occurred: {e}")"""

c3 = new_code_cell(cell3_code)
nb.cells.append(c3)

# Code cell 4: Prompt 2 execution
cell4_code = """def generate():
  client = genai.Client(
        vertexai=True, 
        project=PROJECT_ID, 
        location="global"
    )

  msg1_text1 = types.Part.from_text(text=\"\"\"<text>List of Google Search Results with classified and summarised HTMLs {title: March 20 ChatGPT outage: Here\\'s what happened - OpenAI, snippet: Mar 24, 2023 ... March 20 ChatGPT outage: Here\\'s what happened. An update on our ... 
  This bug only appeared in the Asyncio redis-py client for Redis Cluster, and ..., long_description: An update on our findings, the actions we’ve taken, and technical details of the bug., 
  url: https://openai.com/index/march-20-chatgpt-outage/, sum: An official OpenAI report confirmed that a major ChatGPT outage on March 20, 2023, was caused by a critical bug in the Asyncio redis-py client for Redis Cluster.}{title: Redis Acquires Featureform to Help Developers Deliver, snippet: Oct 9, 2025 ... 
  The acquisition helps Redis solve one of the most critical challenges developers face with production AI: getting structured data into models ..., long_description: Featureform’s powerful framework will add rich structured context to Redis’ fast vector search to deliver the right context to agents at the right time..., 
  url: https://www.globenewswire.com/news-release/2025/10/09/3164211/0/en/redis-acquires-featureform-to-help-developers-deliver-real-time-structured-data-into-ai-agents.html, sum: Redis has announced the acquisition of Featureform, a framework for managing and orchestrating structured data signals, to enhance its real-time data platform for AI agents.}{title: Fenwick Represents Redis in Acquisition of Decodable, snippet: Sep 4, 2025 ... This acquisition will be a strategic step forward in Redis\\'s mission to be the fastest real-time data platform. More information can be obtained ..., long_description: Fenwick is representing Redis Inc., a developer of an open-source in-memory data structure platform designed to be used as a database, cache and message broke, in its acquisition of Decodable, a real-time data platform that lets organizations quickly build, process and manage streaming pipelines., url: Fenwick Represents Redis in Acquisition of Decodable | Fenwick, sum: Redis has announced its acquisition of Decodable, a real-time data platform, introducing potential integration risks associated with M&A activity.}<entity>Redis</entity></text>\"\"\")

  model = "gemini-3.5-flash"
  contents = [
    types.Content(
      role="user",
      parts=[msg1_text1]
    ),
  ]
  tools = [
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(google_maps=types.GoogleMaps()),
  ]
  tool_config = types.ToolConfig(
      retrieval_config = types.RetrievalConfig(),
  )

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    max_output_tokens = 65535,
    safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ],
    tools = tools,
    tool_config = tool_config,
    thinking_config=types.ThinkingConfig(
      thinking_level="High",
    ),
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
        continue
    if chunk.text:
        print(chunk.text, end="")

generate()"""

c4 = new_code_cell(cell4_code)
nb.cells.append(c4)

notebook_filename = "/home/jupyter/prompt_issues/supplier_news_summarization.ipynb"
with open(notebook_filename, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Updated notebook saved to {notebook_filename}")
