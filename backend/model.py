import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from config import api_key

class LLM:
  def __init__(self):
      self.client = OpenAI(
        api_key=api_key,
        base_url="https://llm.chutes.ai/v1"
      )

      self.embedding_model = SentenceTransformer(
          "BAAI/bge-base-en-v1.5"
      )

  def generate(self, prompt, max_tokens=4096, temperature=0.1, json_mode=False):
    params = {
    "model": "Qwen/Qwen3-32B-TEE",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "max_tokens": max_tokens,
    "temperature": temperature
    }

    if json_mode:
        params["response_format"] = {
            "type": "json_object"
        }

    response = self.client.chat.completions.create(**params)

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        response.choices[0].message.content,
        flags=re.DOTALL
    )

    return cleaned.strip()

  def embedding(self, content_to_embed):
      return self.embedding_model.encode(content_to_embed).tolist()

llm = LLM()
