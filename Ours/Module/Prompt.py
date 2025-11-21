# #%% /home/dilab/Desktop/Model/Module/Prompt.py

# ONLY_LLM_WITH_CAPTION_PROMPT = """
# You are a helpful assistant for video question answering (Video QA).

# Question: {question}

# Choices:
# a: {a}
# b: {b}
# c: {c}
# d: {d}
# e: {e}

# Task:
# Based on the video captions, select the most appropriate choice (a, b, c, d, or e).
# Do NOT provide any explanation.
# Answer:
# """

# RAG_WITH_RATIONALE_PROMPT = """
# You are a helpful assistant for video question answering (Video QA).

# Question: {question}

# Choices:
# a: {a}
# b: {b}
# c: {c}
# d: {d}
# e: {e}

# Context from external knowledge:
# {context}

# Video Rationale:
# {rationale}

# Task:
# Using both the context and the video rationale, select the most appropriate choice (a, b, c, d, or e).
# Do NOT provide any explanation. 
# Answer:
# """

# # Ours w/o RAG
# RATIONALE_ONLY_PROMPT = """
# You are a helpful assistant for video question answering (Video QA).

# Question: {question}

# Choices:
# a: {a}
# b: {b}
# c: {c}
# d: {d}
# e: {e}

# Video Rationale:
# {rationale}

# Video Captions:
# {captions}

# Task:
# Based on the video rationale alone, select the most appropriate choice (a, b, c, d, or e).
# Do NOT provide any explanation. 
# Answer:
# """

# # Ours w/o Rationale
# RAG_WITH_CAPTION_PROMPT = """
# You are a helpful assistant for video question answering (Video QA).

# Question: {question}

# Choices:
# a: {a}
# b: {b}
# c: {c}
# d: {d}
# e: {e}

# Context from external knowledge:
# {context}

# Video Captions:
# {captions}

# Task:
# Using both the context and the video captions, select the most appropriate choice (a, b, c, d, or e).
# Do NOT provide any explanation. 
# Answer:
# """
#%% /home/dilab/Desktop/Model/Module/Prompt.py

ONLY_LLM_WITH_CAPTION_PROMPT = """
You are a helpful assistant for video question answering (Video QA).

Question: {question}

Choices:
a: {a}
b: {b}
c: {c}
d: {d}

Task:
Based on the video captions, select the most appropriate choice (a, b, c, d).
Do NOT provide any explanation.
Answer:
"""

RAG_WITH_RATIONALE_PROMPT = """
You are a helpful assistant for video question answering (Video QA).

Question: {question}

Choices:
a: {a}
b: {b}
c: {c}
d: {d}


Context from external knowledge:
{context}

Video Rationale:
{rationale}

Video Captions:
{captions}

Task:
Using both the context and the video rationale, select the most appropriate choice (a, b, c, d).
Do NOT provide any explanation. 
Answer:
"""

# 3️⃣ Ours w/o RAG
RATIONALE_ONLY_PROMPT = """
You are a helpful assistant for video question answering (Video QA).

Question: {question}

Choices:
a: {a}
b: {b}
c: {c}
d: {d}

Video Rationale:
{rationale}

Video Captions:
{captions}

Task:
Based on the video rationale alone, select the most appropriate choice (a, b, c, d).
Do NOT provide any explanation. 
Answer:
"""

# 4️⃣ Ours w/o Rationale
RAG_WITH_CAPTION_PROMPT = """
You are a helpful assistant for video question answering (Video QA).

Question: {question}

Choices:
a: {a}
b: {b}
c: {c}
d: {d}

Context from external knowledge:
{context}

Video Captions:
{captions}

Task:
Using both the context and the video captions, select the most appropriate choice (a, b, c, d).
Do NOT provide any explanation. 
Answer:
"""
