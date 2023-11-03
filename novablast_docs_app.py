import os
import streamlit as st
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from azure.storage.blob import BlobServiceClient
import pickle
import tempfile

st.set_page_config(page_title="NovaBlast: Ask your Blasting question", page_icon="🦜")
image = Image.open('novablast_logo.png')

st.image(image)
st.title("Ask your Blasting question")


@st.cache_resource(ttl="1h")

def retrieve_documents(connection_string):

    service = BlobServiceClient.from_connection_string(connection_string)
    container_client = service.get_container_client("streamlit-docs")

    temp_dir = tempfile.mkdtemp()

    # Get files from Blob
    for blob in container_client.list_blobs():
        download_file_path = os.path.join(temp_dir, blob.name)
        with open(file=download_file_path, mode="wb") as download_file:
            download_file.write(container_client.download_blob(blob.name).readall())

    return temp_dir

def configure_retriever(temp_dir):

    # Load the pdfs from the documents directory
    loader = PyPDFDirectoryLoader("tmp/")
    docs = loader.load()

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    # Create an embeddings db
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    retriever = VectorStoreRetriever(vectorstore=db, search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

connection_string = st.secrets['BLOB_CONNECTION_STRING']

temp_dir = retrieve_documents(connection_string)

openai_api_key = st.secrets['OPENAI_API_KEY']

retriever = configure_retriever(temp_dir)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True #, openai_api_key=openai_api_key
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
