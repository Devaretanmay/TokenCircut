"""OpenAI client wrapper with loop detection."""

from openai import OpenAI
from tokencircuit import TokenCircuitClient

raw = OpenAI()
client = TokenCircuitClient(raw)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
