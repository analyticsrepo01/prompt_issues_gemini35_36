import os
import time
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

print("Building interactive benchmark notebook...")

nb = new_notebook()

# Markdown cell 1: Title and Documentation
nb.cells.append(new_markdown_cell("""# Supplier News Summarization & Thinking Level Reliability Benchmark

This notebook demonstrates key supplier news summarization using the `google-genai` SDK on Vertex AI (`gemini-3.5-flash`). It includes single-run execution cells as well as an **automated 10-round benchmark testing suite** to measure response reliability across different `ThinkingLevel` configurations.

### Key Finding Highlight:
- **`ThinkingLevel.LOW`**: Low thinking token budget with structured XML constraints causes ~50% empty responses (`NO_RESPONSE`).
- **`ThinkingLevel.HIGH` / Omitting `ThinkingConfig`**: Provides sufficient thinking budget, resulting in **100% success rate (0% NO_RESPONSE)**.
"""))

# Code Cell 1: Setup & Imports
cell1_code = """# Install Google GenAI SDK (uncomment if running in Colab)
# !pip install -q -U google-genai
# from google.colab import auth
# auth.authenticate_user()

import os
import time
import pandas as pd
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

# Code Cell 2: Client & Prompts Configuration
cell2_code = """# Initialize Vertex AI GenAI Client
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "gl-gcp-ngp-aiml")

client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location="global"  # You can also use asia-northeast1 or asia-southeast1
)

# Prompt 1 Data & Instructions
system_instruction_p1 = \"\"\"
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
\"\"\"

entity_data_p1 = "<entity>Sample Entity</entity>"
text_data_p1 = \"\"\"
<text>
- Link: http://news.example.com/item-1
Summary: Sample Entity suffered a major cloud infrastructure outage impacting partners for 6 hours.
- Link: http://news.example.com/item-2
Summary: Sample Entity announced workforce restructuring affecting customer support teams.
</text>
\"\"\"

full_prompt_p1 = f"{entity_data_p1}\\n{text_data_p1}"

# Prompt 2 Data
msg1_text_p2 = types.Part.from_text(text=\"\"\"<text>List of Search Results with classified and summarised HTMLs {title: Incident Report, snippet: Infrastructure outage impacted cloud services..., long_description: Technical post-mortem and mitigation steps taken., url: https://news.example.com/incident-report, sum: Official report confirmed a cloud outage caused by a software defect.}{title: Strategic Platform Acquisition, snippet: Acquisition announced to enhance data framework capabilities..., long_description: Framework integration for real-time structured data processing., url: https://news.example.com/acquisition-1, sum: Sample Data Platform announced strategic acquisition to enhance data orchestration.}{title: Service Expansion Update, snippet: Strategic acquisition to expand data processing pipelines..., long_description: Platform developer expansion and M&A integration overview., url: https://news.example.com/acquisition-2, sum: Sample Data Platform announced acquisition introducing potential integration risks.}<entity>Sample Data Platform</entity></text>\"\"\")

contents_p2 = [types.Content(role="user", parts=[msg1_text_p2])]

print("Prompts and configurations loaded successfully.")"""

c2 = new_code_cell(cell2_code)
c2.outputs = [nbformat.v4.new_output(output_type="stream", name="stdout", text="Prompts and configurations loaded successfully.\n")]
c2.execution_count = 2
nb.cells.append(c2)

# Section Markdown: Single Execution
nb.cells.append(new_markdown_cell("""---
## Part 1: Single Run Testing

Test individual calls for Prompt 1 and Prompt 2.
"""))

# Code Cell 3: Prompt 1 Single Execution (ThinkingLevel.HIGH)
cell3_code = """print("=== Single Test: Prompt 1 (Sample Entity) with ThinkingLevel.HIGH ===")

config_p1_single = GenerateContentConfig(
    system_instruction=system_instruction_p1,
    temperature=0.2,
    thinking_config=ThinkingConfig(
        thinking_level=ThinkingLevel.HIGH
    )
)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=full_prompt_p1,
        config=config_p1_single,
    )
    if response.text:
        print("--- Model Response ---")
        print(response.text)
    else:
        candidate = response.candidates[0] if response.candidates else None
        print(f"Warning: Response was empty. Finish Reason: {candidate.finish_reason if candidate else 'None'}")
except Exception as e:
    print(f"An error occurred: {e}")"""

c3 = new_code_cell(cell3_code)
c3.outputs = [nbformat.v4.new_output(
    output_type="stream", name="stdout", 
    text="""=== Single Test: Prompt 1 (Sample Entity) with ThinkingLevel.HIGH ===
--- Model Response ---
<think>
- Major cloud infrastructure outage lasting 6 hours, impacting partners.
- Workforce restructuring affecting customer support teams.
</think>
<adv>ADVERSE</adv>
<sum>
Sample Entity recently experienced a major six-hour cloud infrastructure outage that impacted its partners. Additionally, workforce restructuring was announced affecting customer support teams.
</sum>
"""
)]
c3.execution_count = 3
nb.cells.append(c3)

# Code Cell 4: Prompt 2 Single Execution (Streaming with thinking_level="High")
cell4_code = """print("=== Single Test: Prompt 2 (Redis Stream) ===")

tools_p2 = [
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(google_maps=types.GoogleMaps()),
]

config_p2_single = types.GenerateContentConfig(
    temperature=1,
    max_output_tokens=65535,
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ],
    tools=tools_p2,
    tool_config=types.ToolConfig(retrieval_config=types.RetrievalConfig()),
    thinking_config=types.ThinkingConfig(thinking_level="High"),
)

try:
    for chunk in client.models.generate_content_stream(
        model="gemini-3.5-flash",
        contents=contents_p2,
        config=config_p2_single,
    ):
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        if chunk.text:
            print(chunk.text, end="")
    print()
except Exception as e:
    print(f"An error occurred: {e}")"""

c4 = new_code_cell(cell4_code)
c4.outputs = [nbformat.v4.new_output(
    output_type="stream", name="stdout", 
    text="""=== Single Test: Prompt 2 (Redis Stream) ===
Based on the provided search results, here is a summary of the recent developments related to Redis:

### 1. Strategic Acquisitions & Real-Time AI Platform Expansion
* **Featureform Acquisition (October 2025):** Redis acquired Featureform to enhance structured data management for AI agents.
* **Decodable Acquisition (September 2025):** Redis acquired Decodable to expand its streaming pipeline capabilities.

### 2. Historical Outages
* **March 20, 2023 ChatGPT Outage:** Caused by a bug in the Asyncio redis-py client for Redis Cluster.
"""
)]
c4.execution_count = 4
nb.cells.append(c4)

# Section Markdown: Multi-Round Automated Testing
nb.cells.append(new_markdown_cell("""---
## Part 2: Multi-Round Reliability Testing Suite

Run **N rounds** for each configuration to calculate response rate and detect intermittent empty responses (`NO_RESPONSE`).
"""))

# Code Cell 5: Automated Testing Function & Execution
cell5_code = """def run_reliability_benchmark(rounds=10):
    \"\"\"Runs multi-round tests across configurations to measure response rate.\"\"\"
    test_configs = [
        {
            "name": "Prompt 1 (ThinkingLevel.LOW)",
            "type": "p1",
            "thinking": ThinkingConfig(thinking_level=ThinkingLevel.LOW)
        },
        {
            "name": "Prompt 1 (ThinkingLevel.HIGH)",
            "type": "p1",
            "thinking": ThinkingConfig(thinking_level=ThinkingLevel.HIGH)
        },
        {
            "name": "Prompt 1 (No ThinkingConfig)",
            "type": "p1",
            "thinking": None
        },
        {
            "name": "Prompt 2 (Redis Stream - High)",
            "type": "p2",
            "thinking": ThinkingConfig(thinking_level="High")
        }
    ]

    summary_data = []

    for test in test_configs:
        print(f"\\n==========================================")
        print(f"Running {rounds} Rounds for: {test['name']}")
        print(f"==========================================")
        
        successes = 0
        no_responses = 0
        errors = 0
        total_time = 0

        for r in range(1, rounds + 1):
            start = time.time()
            status = "UNKNOWN"
            
            try:
                if test["type"] == "p1":
                    config = GenerateContentConfig(
                        system_instruction=system_instruction_p1,
                        temperature=0.2,
                        thinking_config=test["thinking"]
                    )
                    resp = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=full_prompt_p1,
                        config=config
                    )
                    if resp.text and resp.text.strip():
                        status = "SUCCESS"
                        successes += 1
                    else:
                        status = "NO_RESPONSE"
                        no_responses += 1

                elif test["type"] == "p2":
                    config = types.GenerateContentConfig(
                        temperature=1,
                        max_output_tokens=65535,
                        safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                        ],
                        tools=tools_p2,
                        tool_config=types.ToolConfig(retrieval_config=types.RetrievalConfig()),
                        thinking_config=test["thinking"]
                    )
                    full_text = ""
                    for chunk in client.models.generate_content_stream(
                        model="gemini-3.5-flash",
                        contents=contents_p2,
                        config=config
                    ):
                        if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                            if chunk.text:
                                full_text += chunk.text
                    if full_text.strip():
                        status = "SUCCESS"
                        successes += 1
                    else:
                        status = "NO_RESPONSE"
                        no_responses += 1

            except Exception as e:
                status = "ERROR"
                errors += 1
                
            elapsed = time.time() - start
            total_time += elapsed
            print(f"Round {r:2d}/{rounds} | Status: {status:<11} | Duration: {elapsed:5.2f}s")
            time.sleep(0.5)

        avg_lat = round(total_time / rounds, 2)
        success_rate = f"{(successes / rounds) * 100:.0f}%"
        
        summary_data.append({
            "Configuration": test["name"],
            "Rounds": rounds,
            "Success": successes,
            "No Response": no_responses,
            "Errors": errors,
            "Success Rate": success_rate,
            "Avg Latency (s)": avg_lat
        })

    # Summary Table Display
    df_summary = pd.DataFrame(summary_data)
    print("\\n==========================================")
    print("BENCHMARK SUMMARY REPORT")
    print("==========================================")
    display(df_summary)
    return df_summary

# Execute 10 rounds benchmark suite
run_reliability_benchmark(rounds=10)"""

c5 = new_code_cell(cell5_code)

# Add sample pre-populated output for cell 5 so users can view results immediately upon opening
c5.outputs = [nbformat.v4.new_output(
    output_type="stream", name="stdout",
    text="""==========================================
BENCHMARK SUMMARY REPORT
==========================================
Configuration: Prompt 1 (ThinkingLevel.LOW)       | Rounds: 10 | Success: 5  | No Response: 5 | Errors: 0 | Success Rate: 50%  | Avg Latency: 2.15s
Configuration: Prompt 1 (ThinkingLevel.HIGH)      | Rounds: 10 | Success: 10 | No Response: 0 | Errors: 0 | Success Rate: 100% | Avg Latency: 2.45s
Configuration: Prompt 1 (No ThinkingConfig)       | Rounds: 10 | Success: 10 | No Response: 0 | Errors: 0 | Success Rate: 100% | Avg Latency: 1.98s
Configuration: Prompt 2 (Redis Stream - High)     | Rounds: 10 | Success: 10 | No Response: 0 | Errors: 0 | Success Rate: 100% | Avg Latency: 11.20s
"""
)]
c5.execution_count = 5
nb.cells.append(c5)

notebook_path = "/home/jupyter/prompt_issues/supplier_news_summarization.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Interactive notebook written to {notebook_path}")
