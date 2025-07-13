from agno.agent import Agent
from agno.models.openai import OpenAIChat
import streamlit as st
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from knowledge_base import KnowledgeWorkerThread
from ebird_tool import get_sightings, get_region_codes
from textwrap import dedent
import uuid


@st.cache_resource
def get_knowledge_base():
    thread = KnowledgeWorkerThread("ebird_json")
    thread.start()
    thread.join()  # Wait for the thread to finish
    if thread.error:
        st.error(f"Error loading knowledge base: {thread.error}")
        return None
    return thread.knowledge_base


# Initialize session state for session_id and messages
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

memory_db = SqliteMemoryDb(table_name="memory", db_file="memory.db")
memory = Memory(
    db=memory_db,
    delete_memories=False,
    clear_memories=False,
    summarize=True,  # Enable summarization
)
storage = SqliteStorage(table_name="agent_sessions", db_file="memory.db")

knowledge_base = get_knowledge_base()

if knowledge_base:
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        knowledge=knowledge_base,
        memory=memory,
        session_id=st.session_state.session_id,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_runs=4,  # Reduce history length
        knowledge_max_results=3,  # Reduce knowledge results
        markdown=True,
        description="You are an expert ornithologist, that knows all about birds, bird classification and bird biology. You will help the user learn about birds, and guide and customize their learning to their individual needs.",
        instructions=dedent(
            """\
        When asked for sightings on a bird in a location
        First look up the region code for the state (e.g. TX)
        using the provided tools. If the state is not provided,
        try to figure out the state code from the given region.
        Then look up the bird species code by common name
        in the knowledge base. Finally, get the sightings
        for the given species and region using the provided tool.
        Summarize the results in a readable format.
        """
        ),
        tools=[get_sightings, get_region_codes],
        storage=storage,
        read_tool_call_history=True,
        search_knowledge=True,
    )

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Let's learn about birds!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = agent.run(prompt)
            st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
