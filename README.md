# Gemini 3.5 Flash Thinking Level Reliability Benchmark

A systematic benchmark and analysis suite testing the response reliability of `gemini-3.5-flash` on **Google Cloud Vertex AI** when combining strict XML formatting constraints with different `ThinkingConfig` (`ThinkingLevel`) settings.

---

## 📌 Executive Summary & Key Findings

When generating responses with `gemini-3.5-flash`, the model allocates reasoning/thinking tokens prior to generating final output text. Our empirical benchmarks across multiple test suites reveal key insights regarding thinking token allocation and output reliability:

### **Benchmark Comparison Table**

| Prompt Scenario | Formatting / Interaction | `ThinkingLevel.LOW` | `ThinkingLevel.HIGH` | No `ThinkingConfig` (Default) |
| :--- | :--- | :---: | :---: | :---: |
| **Prompt 1 (Supplier News)** | Unary + Strict XML Tags (`<think>`, `<adv>`, `<sum>`) | ⚠️ **~50% NO_RESPONSE** | ✅ **100% SUCCESS** | ✅ **100% SUCCESS** |
| **Prompt 2 (Redis News)** | Stream + Tools (`GoogleSearch`, `GoogleMaps`) | ✅ **100% SUCCESS** | ✅ **100% SUCCESS** | ✅ **100% SUCCESS** |

---

## 🔍 Root Cause Analysis: The `ThinkingLevel.LOW` Empty Response Issue

1. **Token Exhaustion During Reasoning**: Under `ThinkingLevel.LOW`, the model is restricted to a small thinking token budget. When required to follow strict structural output constraints (such as pre-defined XML schema tags like `<think>`, `<adv>`, `<sum>`), the model consumes its entire allocated token budget inside internal thinking before it can produce the initial text tokens.
2. **FinishReason / NO_RESPONSE**: As a result, the model finishes execution without generating visible response content, returning an empty response (`NO_RESPONSE` / `FinishReason.STOP`).
3. **The Solution (`ThinkingLevel.HIGH` or Default)**: Setting `thinking_level=ThinkingLevel.HIGH` or omitting `ThinkingConfig` allows the model sufficient reasoning bandwidth to process formatting instructions before producing output, achieving **100% response reliability**.

---

## 📂 Repository Layout

```
.
├── README.md                                 # Project documentation & benchmark overview
├── gemini_thinking_reliability_benchmark.ipynb# 100% self-contained Jupyter benchmark notebook
├── test_prompt1_configs.py                  # Standalone test runner for Prompt 1 (XML Schema)
├── test_prompt2_configs.py                  # Standalone test runner for Prompt 2 (Tools & Streaming)
├── create_clean_notebook.py                 # Generator script to rebuild the self-contained notebook
├── run_benchmark.py                         # Unified benchmark runner across all prompts
└── .gitignore                               # Git ignore rules (.ipynb_checkpoints, etc.)
```

---

## 🚀 Quick Start & How to Run

### **Prerequisites**

Ensure you have Python 3.10+ installed along with the official `google-genai` SDK and `pandas`:

```bash
pip install -q -U google-genai pandas
```

### **1. Running via Jupyter Notebook**

Open [`gemini_thinking_reliability_benchmark.ipynb`](./gemini_thinking_reliability_benchmark.ipynb) in Jupyter, Google Colab, or Vertex AI Workbench. 

The notebook is **100% self-contained** and includes:
- **Step 1**: SDK setup & imports.
- **Step 2**: Vertex AI Client initialization.
- **Step 3**: Single-call verification for both Prompt 1 and Prompt 2.
- **Step 4**: Automated 10-round benchmark runner (`run_reliability_benchmark(rounds=10)`) that executes across all test configurations and prints a structured summary DataFrame.

### **2. Running Standalone Benchmark Scripts**

Set your GCP project environment variable and run the standalone scripts directly from your terminal:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# Test Prompt 1 (Strict XML Schema Format)
python3 test_prompt1_configs.py

# Test Prompt 2 (Streaming + Search & Maps Tools)
python3 test_prompt2_configs.py

# Run Full Unified Benchmark
python3 run_benchmark.py
```

---

## ⚙️ Code Example & Best Practices

When initializing `GenerateContentConfig` for structured output tasks in `gemini-3.5-flash`:

```python
import os
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, ThinkingLevel

# Initialize Vertex AI Client
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")

# Recommended Config for Strict Formatting / Schema Outputs:
config = GenerateContentConfig(
    system_instruction="Your system instruction with strict XML/JSON formatting...",
    temperature=0.2,
    thinking_config=ThinkingConfig(
        thinking_level=ThinkingLevel.HIGH  # Recommended to avoid NO_RESPONSE
    )
)

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Your input text...",
    config=config
)

print(response.text)
```

---

## 💡 Recommendations for Developers

1. **For Strict Schema Constraints (XML / JSON)**: Always use `ThinkingLevel.HIGH` or leave `thinking_config` unconfigured (default). Avoid `ThinkingLevel.LOW` as it risks high `NO_RESPONSE` rates.
2. **For Tool Use & Streaming**: Both `LOW` and `HIGH` thinking levels work reliably without response drops.
3. **Multi-Region Location Support**: Supported Vertex AI global locations include `global`, `us-central1`, `asia-northeast1`, and `asia-southeast1`.

---

## 📄 License

This repository is provided for benchmarking and educational purposes under the MIT License.
