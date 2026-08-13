import os
import sys
import time
import traceback
from google import genai
from google.genai import types
from google.genai.types import (
    GenerateContentConfig,
    ThinkingConfig,
    ThinkingLevel,
)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project-0004-346516")

print(f"Running Benchmark with PROJECT_ID={PROJECT_ID}")

# Prompt 1 setup
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
full_prompt_1 = f"{entity_data}\n{text_data}"


# Prompt 2 setup
msg1_text1 = types.Part.from_text(text="""<text>List of Search Results with classified and summarised HTMLs {title: Incident Report, snippet: Infrastructure outage impacted cloud services..., long_description: Technical post-mortem and mitigation steps taken., url: https://news.example.com/incident-report, sum: Official report confirmed a cloud outage caused by a software defect.}{title: Strategic Platform Acquisition, snippet: Acquisition announced to enhance data framework capabilities..., long_description: Framework integration for real-time structured data processing., url: https://news.example.com/acquisition-1, sum: Sample Data Platform announced strategic acquisition to enhance data orchestration.}{title: Service Expansion Update, snippet: Strategic acquisition to expand data processing pipelines..., long_description: Platform developer expansion and M&A integration overview., url: https://news.example.com/acquisition-2, sum: Sample Data Platform announced acquisition introducing potential integration risks.}<entity>Sample Data Platform</entity></text>""")

contents_2 = [
    types.Content(
        role="user",
        parts=[msg1_text1]
    ),
]
tools_2 = [
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(google_maps=types.GoogleMaps()),
]
tool_config_2 = types.ToolConfig(
    retrieval_config=types.RetrievalConfig(),
)

# Test both location="global" and location="asia-northeast1" if needed, but let's test location="global" first as specified in user's notebook code
locations = ["global", "asia-northeast1"]

results_prompt_1 = []
results_prompt_2 = []

print("\n==========================================")
print("STARTING 10 ROUNDS FOR PROMPT 1 (Sample Entity)")
print("==========================================")

client_p1 = genai.Client(vertexai=True, project=PROJECT_ID, location="global")

config_p1 = GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.2,
    thinking_config=ThinkingConfig(
        thinking_level=ThinkingLevel.LOW
    )
)

for i in range(1, 11):
    start_time = time.time()
    status = "UNKNOWN"
    text_content = ""
    error_msg = None
    finish_reason = None
    
    try:
        response = client_p1.models.generate_content(
            model="gemini-3.5-flash",
            contents=full_prompt_1,
            config=config_p1,
        )
        duration = round(time.time() - start_time, 2)
        
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason
            
        if response.text and response.text.strip():
            status = "SUCCESS"
            text_content = response.text
        else:
            status = "NO_RESPONSE"
            
    except Exception as e:
        duration = round(time.time() - start_time, 2)
        status = "ERROR"
        error_msg = str(e)
        
    results_prompt_1.append({
        "round": i,
        "status": status,
        "duration": duration,
        "finish_reason": finish_reason,
        "text_length": len(text_content),
        "error": error_msg
    })
    print(f"Prompt 1 | Round {i:2d}/10 | Status: {status:<11} | Time: {duration:5.2f}s | FinishReason: {finish_reason} | TextLen: {len(text_content)} | Error: {error_msg}")
    time.sleep(1)


print("\n==========================================")
print("STARTING 10 ROUNDS FOR PROMPT 2 (Sample Data Platform Stream)")
print("==========================================")

client_p2 = genai.Client(vertexai=True, project=PROJECT_ID, location="global")

config_p2 = types.GenerateContentConfig(
    temperature=1,
    max_output_tokens=65535,
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ],
    tools=tools_2,
    tool_config=tool_config_2,
    thinking_config=types.ThinkingConfig(
        thinking_level="High",
    ),
)

for i in range(1, 11):
    start_time = time.time()
    status = "UNKNOWN"
    text_chunks = []
    error_msg = None
    chunks_count = 0
    candidate_parts_seen = 0
    
    try:
        stream = client_p2.models.generate_content_stream(
            model="gemini-3.5-flash",
            contents=contents_2,
            config=config_p2,
        )
        for chunk in stream:
            chunks_count += 1
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            candidate_parts_seen += len(chunk.candidates[0].content.parts)
            # Inspect chunk text
            if chunk.text:
                text_chunks.append(chunk.text)
                
        duration = round(time.time() - start_time, 2)
        full_text = "".join(text_chunks)
        
        if full_text.strip():
            status = "SUCCESS"
        else:
            status = "NO_RESPONSE"
            
    except Exception as e:
        duration = round(time.time() - start_time, 2)
        status = "ERROR"
        error_msg = str(e)
        
    full_text = "".join(text_chunks)
    results_prompt_2.append({
        "round": i,
        "status": status,
        "duration": duration,
        "chunks": chunks_count,
        "parts_seen": candidate_parts_seen,
        "text_length": len(full_text),
        "error": error_msg
    })
    print(f"Prompt 2 | Round {i:2d}/10 | Status: {status:<11} | Time: {duration:5.2f}s | Chunks: {chunks_count:2d} | Parts: {candidate_parts_seen:2d} | TextLen: {len(full_text)} | Error: {error_msg}")
    time.sleep(1)

print("\n==========================================")
print("SUMMARY OF BENCHMARK RESULTS")
print("==========================================")

p1_success = sum(1 for r in results_prompt_1 if r["status"] == "SUCCESS")
p1_no_resp = sum(1 for r in results_prompt_1 if r["status"] == "NO_RESPONSE")
p1_errors  = sum(1 for r in results_prompt_1 if r["status"] == "ERROR")

p2_success = sum(1 for r in results_prompt_2 if r["status"] == "SUCCESS")
p2_no_resp = sum(1 for r in results_prompt_2 if r["status"] == "NO_RESPONSE")
p2_errors  = sum(1 for r in results_prompt_2 if r["status"] == "ERROR")

print(f"Prompt 1 (Sample Entity):               {p1_success}/10 SUCCESS | {p1_no_resp}/10 NO_RESPONSE | {p1_errors}/10 ERROR")
print(f"Prompt 2 (Sample Data Platform Stream): {p2_success}/10 SUCCESS | {p2_no_resp}/10 NO_RESPONSE | {p2_errors}/10 ERROR")
