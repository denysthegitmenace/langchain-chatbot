import langchain
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import (ConversationBufferMemory,
                              ConversationSummaryMemory)
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)
from langchain.vectorstores import Chroma

langchain.debug = True


llm = ChatOpenAI()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

conversation(
    {"question": "Translate this sentence from English to French: I love programming."}
)


# custom knowledge
loader = PyPDFLoader("private data5.pdf")
pages = loader.load_and_split()


vectorstore = Chroma.from_documents(documents=pages, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)


qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
