from dotenv import load_dotenv
import os

load_dotenv()
print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
# optional, show only the first characters
key = os.getenv("OPENAI_API_KEY")
if key:
    print("OPENAI_API_KEY preview:", key[:8] + "...")