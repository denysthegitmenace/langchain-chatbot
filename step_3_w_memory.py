from pprint import pprint

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import (ConversationBufferMemory,
                              ConversationSummaryMemory)

chat = ChatOpenAI()
conversation = ConversationChain(llm=chat)

# TODO change fake responses to the real ones
memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")
memory.save_context(
    {"input": "how are you doing today"}, {"output": "fine thank you. and you?"}
)
memory.save_context(
    {"input": "very well thank you."}, {"output": "how can I help you today?"}
)

pprint(memory.load_memory_variables({}))


llm = OpenAI()
memory = ConversationSummaryMemory(llm=llm)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context(
    {"input": "how are you doing today"}, {"output": "fine thank you. and you?"}
)
memory.save_context(
    {"input": "very well thank you."}, {"output": "how can I help you today?"}
)
pprint(memory.load_memory_variables({}))
