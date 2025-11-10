 Vibe Matcher – Mini Fashion Recommender

**Author:** Shruti Kulkarni  
**Context:** Prototype built for Nexora’s AI product matching challenge Concept
Input a *vibe query* like “energetic urban chic”.  
Each product description is embedded with `text-embedding-ada-002`.  
We compute cosine similarity to rank and return top-3 items.

 Quick Start
1. Clone or open in Google Colab  
2. Add your OpenAI key:
   import os
   os.environ["Enter your OPENAI_API_KEY"] = "xxxxxxxx"

