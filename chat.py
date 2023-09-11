import langchain
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space

langchain.debug = True
st.set_page_config(page_title="OpenAI-powered Langchain ChatBot")

# Sidebar contents
with st.sidebar:
    st.title("🦜⛓️🤖 OpenAI-powered Langchain ChatBot")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [LangChain](https://www.langchain.com/)
    - [OpenAI ChatGPT](https://chat.openai.com)
    - [Streamlit](https://streamlit.io/)

    💡 Note: Your OpenAI API key must be defined under OPENAI_API_KEY env var!
    """
    )
    add_vertical_space(5)
    st.write("Made with ❤️ by Denys on data")

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

chatGPT = ChatOpenAI()

info_about_berlin_loader = TextLoader("real-time-information-on-berlin.txt")

vectorstore = Chroma.from_documents(
    documents=info_about_berlin_loader.load(), embedding=OpenAIEmbeddings()
)


template = """System: Use the following pieces of context to answer the users question. \nIf you don't know the answer, answer the question once again ignoring the context.\n----------------\n{context}\nHuman: {question}"""
prompt = PromptTemplate(template=template, input_variables=["question", "context"])
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_with_memory_and_knowledge = ConversationalRetrievalChain.from_llm(
    chatGPT,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
)


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()


if user_input:
    result = result = chat_with_memory_and_knowledge({"question": user_input})
    output = f"Answer: {result['answer']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))
