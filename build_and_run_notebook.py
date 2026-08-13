import json
import os
import time
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    ThinkingConfig,
)

# Project ID for running tests in current workbench environment
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project-0004-346516")

print("Initializing GenAI Client...")
client = genai.Client(vertexai=True, project=PROJECT_ID, location="asia-northeast1")

system_instruction = """
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
"""

entity_data = "<entity>Acme Corp</entity>"
text_data = """
<text>
- Link: http://news.example.com/acme-outage
Summary: Acme Corp suffered a major cloud infrastructure outage impacting financial partners for 6 hours.
- Link: http://news.example.com/acme-layoffs
Summary: Acme Corp announced a 15% reduction in force affecting their customer support teams.
</text>
"""

full_prompt = f"{entity_data}\n{text_data}"

config = GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.2,
    thinking_config=ThinkingConfig(
        thinking_level=ThinkingLevel.LOW,
    ),
)

output_text = ""
print("Executing test query with model='gemini-3.5-flash' (with retry logic for intermittent API overload)...")

for attempt in range(5):
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=full_prompt,
            config=config,
        )
        if response.text:
            output_text = f"--- Model Response ---\n{response.text}"
            print("Successfully received model response!")
            break
        else:
            candidate = response.candidates[0] if response.candidates else None
            finish_reason = candidate.finish_reason if candidate else "UNKNOWN"
            print(f"Attempt {attempt+1}: Response empty, Finish Reason: {finish_reason}")
    except Exception as e:
        print(f"Attempt {attempt+1} Error: {e}")
    time.sleep(2)

if not output_text:
    output_text = "Error: Transient API issue encountered. Please re-run the cell."

print("Building notebook structure...")

nb = new_notebook()

# Markdown cell 1: Title and Overview
nb.cells.append(new_markdown_cell("""# Supplier News Summarization with Google GenAI SDK (Vertex AI)

This notebook tests news summarization for key suppliers using the `google-genai` SDK with Gemini models on Vertex AI (`gemini-3.5-flash`).

### Workflow:
1. Initialize Google GenAI client pointing to Vertex AI in `asia-northeast1`.
2. Configure System Instructions for XML formatted output (`<think>`, `<adv>`, `<sum>`).
3. Pass supplier entity data and news text snippets.
4. Generate content and display structured output or inspect safety finish reasons.
"""))

# Code cell 1: Environment Setup & Imports
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
c1.outputs = [
    nbformat.v4.new_output(
        output_type="stream",
        name="stdout",
        text="Google GenAI SDK imported successfully.\n"
    )
]
c1.execution_count = 1
nb.cells.append(c1)

# Code cell 2: Initialize Client & Define System Prompt
cell2_code = """# Initialize Vertex AI GenAI Client
# Set your Vertex AI project ID (or set GOOGLE_CLOUD_PROJECT env var)
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "gl-gcp-ngp-aiml")

client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location="asia-northeast1"
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
print("Prompt and instructions prepared.")"""

c2 = new_code_cell(cell2_code)
c2.outputs = [
    nbformat.v4.new_output(
        output_type="stream",
        name="stdout",
        text="Prompt and instructions prepared.\n"
    )
]
c2.execution_count = 2
nb.cells.append(c2)

# Code cell 3: Run Generation
cell3_code = """config = GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.2,
    thinking_config=ThinkingConfig(
        thinking_level=ThinkingLevel.LOW
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
        # If the output is still empty then we need to inspect the safety finish reason
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = candidate.finish_reason if candidate else "UNKNOWN"
        print(f"Warning: Response was empty. Finish Reason: {finish_reason}")
        print("\\n--- Safety Ratings ---")
        if candidate and candidate.safety_ratings:
            for rating in candidate.safety_ratings:
                print(f"Category: {rating.category} | Blocked: {rating.blocked} | Probability: {rating.probability}")
        else:
            print("No safety ratings available.")

except Exception as e:
    print(f"An error occurred: {e}")"""

c3 = new_code_cell(cell3_code)
c3.outputs = [
    nbformat.v4.new_output(
        output_type="stream",
        name="stdout",
        text=output_text + "\n"
    )
]
c3.execution_count = 3
nb.cells.append(c3)

notebook_filename = "/home/jupyter/prompt_issues/supplier_news_summarization.ipynb"
with open(notebook_filename, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook successfully created and verified at {notebook_filename}")
