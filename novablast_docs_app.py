import os
import streamlit as st
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.prompts.prompt import PromptTemplate

st.set_page_config(page_title="NovaBlast: Ask your Blasting question", page_icon="🦜")
image = Image.open('novablast_logo.png')

st.image(image)
st.title("Ask your Blasting question")

@st.cache_resource(ttl="1h")
def configure_retriever():

    # Set secrets through Streamlit
    # os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"] = st.secrets['AZURE_COGNITIVE_SEARCH_SERVICE_NAME']
    # os.environ["AZURE_COGNITIVE_SEARCH_INDEX_NAME"] = st.secrets['AZURE_COGNITIVE_SEARCH_INDEX_NAME']
    # os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"] = st.secrets['AZURE_COGNITIVE_SEARCH_API_KEY']

    retriever = AzureCognitiveSearchRetriever(content_key="content", top_k=3)

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
            source = os.path.basename(doc.metadata["metadata_storage_path"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

# Set env variables through Streamlit
# os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM
llm = ChatOpenAI(
    model_name="gpt-4", temperature=0, streaming=True
)

# Setup custom prompt template
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
End the standalone question with: "Important: under no circumstances provide any contact details found in the context."
If the follow up question is not a question, then simply repeat the text provided after Follow Up Input.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# Combine LLM and prompt to setup QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, 
    retriever=retriever, 
    # condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    memory=memory,
    verbose=True
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
        # retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[stream_handler])
