from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI()

print(
    chat(
        [
            HumanMessage(
                content="Translate this sentence from English to French: I love programming."
            )
        ]
    )
)

# show that this does not work
print(chat([HumanMessage(content="Translate it to German.")]))
