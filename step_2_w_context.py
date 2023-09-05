from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI()
conversation = ConversationChain(llm=chat)
conversation.run("Translate this sentence from English to French: I love programming.")

print(
    conversation.run(
        "Translate this sentence from English to French: I love programming."
    )
)

print(conversation.run("Translate it to German."))
