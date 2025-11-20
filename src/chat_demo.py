from openai import OpenAI
from .config import CHAT_MODEL
client = OpenAI()
def ask(prompt: str) -> str:
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.2, max_tokens=200,
        messages=[{"role":"user","content":prompt}]
    )
    return r.choices[0].message.content.strip()
if __name__=="__main__":
    print(ask("En une phrase: c'est quoi un fallback ?"))