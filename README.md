# 🤖 LLM Chat Bot using Mistral-7B and llama-cpp-python

This project demonstrates how to deploy and interact with a large language model (LLM) locally using the **Mistral-7B Instruct** model and the `llama-cpp-python` library. The model is run on **Google Colab with a T4 GPU** and performs real-time question answering.

---

## 🚀 Features

- Loads a quantized 4-bit version of Mistral-7B Instruct (`.gguf` format)
- Supports interactive text generation using custom user prompts
- Efficient inference using `llama-cpp-python` backend
- No need for cloud-based APIs — full model runs locally in the Colab environment

---

## 🛠️ Technologies Used

- `llama-cpp-python`
- `Google Colab` with NVIDIA T4 GPU
- `Mistral-7B Instruct v0.1` (Q4_K_M, 4-bit GGUF format)
- Hugging Face model hub (for model download)

---

## 💬 Sample Prompts

```text
What is the capital of India?
What are the names of the 7 wonders of the world?
Write a paragraph about Indian development.
What is the capital of America?
What are the names of all the Continents?
Write a paragraph about Indian Democracy.
Write a paragraph about the current situations of Russia and Ukraine war and the worlds perpective.
What are the names of the aircraft carries of indian navy.
What is the capital of Egypt?
What are the name of the space organization of USA?
What is the name of the most advanced jet fighters of the world?

````

---

## 📥 Setup Instructions

1. Clone the repository or open the notebook in Google Colab.

2. Install dependencies:

```bash
pip install llama-cpp-python
```

3. Download the quantized model:

```bash
!wget -O mistral-7b-instruct.Q4_K_M.gguf \
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

4. Load the model and run queries using:

```python
from llama_cpp import Llama

llm = Llama(model_path="mistral-7b-instruct.Q4_K_M.gguf", n_ctx=2048)

response = llm("[INST] What is the capital of India? [/INST]", max_tokens=256)
print(response['choices'][0]['text'].strip())
```

---

## 📌 Notes

* Make sure to enable GPU in Colab:
  `Runtime > Change runtime type > Hardware accelerator > GPU`

* For better performance, you can configure `n_gpu_layers` and `n_threads` based on your environment.


