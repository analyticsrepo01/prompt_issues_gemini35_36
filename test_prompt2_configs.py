import os
import time
from google import genai
from google.genai import types
from google.genai.types import ThinkingConfig, ThinkingLevel

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "my-project-0004-346516")
client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")

msg1_text_p2 = types.Part.from_text(text="""<text>List of Google Search Results with classified and summarised HTMLs {title: March 20 ChatGPT outage: Here\\'s what happened - OpenAI, snippet: Mar 24, 2023 ... March 20 ChatGPT outage: Here\\'s what happened. An update on our ... 
  This bug only appeared in the Asyncio redis-py client for Redis Cluster, and ..., long_description: An update on our findings, the actions we’ve taken, and technical details of the bug., 
  url: https://openai.com/index/march-20-chatgpt-outage/, sum: An official OpenAI report confirmed that a major ChatGPT outage on March 20, 2023, was caused by a critical bug in the Asyncio redis-py client for Redis Cluster.}{title: Redis Acquires Featureform to Help Developers Deliver, snippet: Oct 9, 2025 ... 
  The acquisition helps Redis solve one of the most critical challenges developers face with production AI: getting structured data into models ..., long_description: Featureform’s powerful framework will add rich structured context to Redis’ fast vector search to deliver the right context to agents at the right time..., 
  url: https://www.globenewswire.com/news-release/2025/10/09/3164211/0/en/redis-acquires-featureform-to-help-developers-deliver-real-time-structured-data-into-ai-agents.html, sum: Redis has announced the acquisition of Featureform, a framework for managing and orchestrating structured data signals, to enhance its real-time data platform for AI agents.}{title: Fenwick Represents Redis in Acquisition of Decodable, snippet: Sep 4, 2025 ... This acquisition will be a strategic step forward in Redis\\'s mission to be the fastest real-time data platform. More information can be obtained ..., long_description: Fenwick is representing Redis Inc., a developer of an open-source in-memory data structure platform designed to be used as a database, cache and message broke, in its acquisition of Decodable, a real-time data platform that lets organizations quickly build, process and manage streaming pipelines., url: Fenwick Represents Redis in Acquisition of Decodable | Fenwick, sum: Redis has announced its acquisition of Decodable, a real-time data platform, introducing potential integration risks associated with M&A activity.}<entity>Redis</entity></text>""")

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


