import os
import time
from google import genai
from google.genai import types
from google.genai.types import ThinkingConfig, ThinkingLevel

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project-0004-346516")
client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")

msg1_text_p2 = types.Part.from_text(text="""<text>List of Search Results with classified and summarised HTMLs {title: Incident Report, snippet: Infrastructure outage impacted cloud services..., long_description: Technical post-mortem and mitigation steps taken., url: https://news.example.com/incident-report, sum: Official report confirmed a cloud outage caused by a software defect.}{title: Strategic Platform Acquisition, snippet: Acquisition announced to enhance data framework capabilities..., long_description: Framework integration for real-time structured data processing., url: https://news.example.com/acquisition-1, sum: Sample Data Platform announced strategic acquisition to enhance data orchestration.}{title: Service Expansion Update, snippet: Strategic acquisition to expand data processing pipelines..., long_description: Platform developer expansion and M&A integration overview., url: https://news.example.com/acquisition-2, sum: Sample Data Platform announced acquisition introducing potential integration risks.}<entity>Sample Data Platform</entity></text>""")

contents_p2 = [types.Content(role="user", parts=[msg1_text_p2])]

tools_p2 = [
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(google_maps=types.GoogleMaps()),
]

def run_prompt2_config_test(name, thinking_cfg):
    print(f"\n==========================================")
    print(f"Testing Prompt 2 Config: {name}")
    print(f"==========================================")
    successes = 0
    no_responses = 0
    errors = 0
    for r in range(1, 11):
        start = time.time()
        try:
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
                thinking_config=thinking_cfg
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
            elapsed = time.time() - start
            if full_text.strip():
                successes += 1
                print(f"Round {r:2d}/10 | SUCCESS | Latency: {elapsed:5.2f}s | Len: {len(full_text)}")
            else:
                no_responses += 1
                print(f"Round {r:2d}/10 | NO_RESPONSE | Latency: {elapsed:5.2f}s")
        except Exception as e:
            errors += 1
            elapsed = time.time() - start
            print(f"Round {r:2d}/10 | ERROR ({e}) | Latency: {elapsed:5.2f}s")
        time.sleep(0.5)
    print(f"Result for Prompt 2 ({name}): {successes}/10 SUCCESS | {no_responses}/10 NO_RESPONSE | {errors}/10 ERROR")

# Run 1: Prompt 2 with ThinkingLevel.LOW
run_prompt2_config_test("ThinkingLevel.LOW", ThinkingConfig(thinking_level=ThinkingLevel.LOW))

# Run 2: Prompt 2 with ThinkingLevel.HIGH
run_prompt2_config_test("ThinkingLevel.HIGH", ThinkingConfig(thinking_level=ThinkingLevel.HIGH))

# Run 3: Prompt 2 without ThinkingConfig
run_prompt2_config_test("No ThinkingConfig", None)


