# routes/chat_routes.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from io import StringIO
import json
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from schema.chat_schemas import ChatMessage, ChatResponse, ChatHistory
from config.settings import settings
from utils.chat_history import JSONChatHistory
from utils.vanna_extension import MyVanna
from logs.logger import chat_logger, sql_logger, workflow_logger
from config.database import get_db_connection, get_llm

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, FewShotChatMessagePromptTemplate
from utils.few_shot import load_few_shot_examples

router = APIRouter()

# Initialize components
json_chat_history = JSONChatHistory(settings.CHAT_HISTORY_PATH)
llm = get_llm(settings.GROQ_API_KEY)
db = get_db_connection(settings.DB_PATH)

# Initialize Vanna
vn = MyVanna(chat_model=llm)
vn.connect_to_sqlite(settings.DB_PATH)

# Set up workflow
max_retries = settings.MAX_SQL_RETRIES

def sql_gen_node(state):
    error = state.get("sql_error")
    previous_sql = state.get("previous_sql")
    prompt = state["rephrased_question"]

    # Check if there's a previous DataFrame in the state (follow-up)
    previous_df_json = state.get("df")
    if previous_df_json:
        try:
            previous_df = pd.read_json(StringIO(previous_df_json), orient='records')
            prompt += f"\nPrevious data (sampled):\n{previous_df.head().to_string()}"
        except Exception as e:
            sql_logger.error(f"Error loading previous DataFrame: {e}")

    if error and previous_sql:
        prompt = f"""SQL Error: {error}
        Failed SQL: {previous_sql}
        Question: {prompt}
        Please generate a corrected SQL query:"""
        state["sql_retries"] = state.get("sql_retries", 0) + 1
        sql_logger.info(f"SQL Error detected, incrementing sql_retries to: {state['sql_retries']}")
    else:
        state["sql_retries"] = 0  # Reset retries when there is no error
        sql_logger.info("No SQL error, resetting sql_retries to 0")

    sql_logger.info(f"Generating SQL with prompt: {prompt}")
    state["sql"] = vn.generate_sql(prompt)
    sql_logger.info(f"Generated SQL: {state['sql']}")

    return state

def execute_sql_node(state):
    try:
        current_sql = state["sql"]
        sql_logger.info(f"Executing SQL: {current_sql}")
        df = vn.run_sql(state['sql'])

        # Check if the DataFrame is empty
        if df.empty:
            sql_logger.info("SQL query returned an empty DataFrame.")
            state['df'] = "[]"  # Store an empty JSON array
        else:
            # Convert the DataFrame to JSON
            try:
                df_json = df.to_json(orient='records', date_format='iso')
                state['df'] = df_json
            except Exception as e:
                sql_logger.error(f"Error converting DataFrame to JSON: {e}")
                state['df'] = "[]"  # Store an empty JSON array on error

        state["sql_error"] = None
        state["previous_sql"] = None
        sql_logger.info(f"SQL executed successfully.")

    except Exception as e:
        state["sql_error"] = str(e)
        state["previous_sql"] = current_sql
        state["df"] = "[]"  # Store an empty JSON array in case of SQL error
        sql_logger.error(f"SQL Error: {str(e)}, SQL: {current_sql}")

    return state

def sql_response_node(state):
    df_json = state['df']

    # Check if df_json is None or empty string
    if not df_json or df_json == "[]":
        workflow_logger.info("No data to process. Returning a default response.")
        state['response'] = "No results found for this query."
        json_chat_history.add_interaction(
            question=state["question"],
            rephrased_question=state.get("rephrased_question"),
            response=state['response'],
            category="sql",
            sql=state.get('sql'),
            sql_error=state.get('sql_error'),
            df=None  # Indicate no data
        )
        return state

    try:
        df = pd.read_json(StringIO(df_json), orient='records')

        # Limit DataFrame size to avoid exceeding token limits
        max_rows = settings.MAX_ROWS_RETURN
        if len(df) > max_rows:
            workflow_logger.info(f"DataFrame has {len(df)} rows. Sampling to {max_rows} rows.")
            df = df.sample(n=max_rows, random_state=42)

        question_for_summary = state.get("rephrased_question", state["question"])
        workflow_logger.info(f"sql_response_node: question_for_summary = {question_for_summary}")
        df_string = df.to_string()
        workflow_logger.info(f"sql_response_node: DataFrame to string length = {len(df_string)}")

        response = vn.generate_summary(question_for_summary, df)
        state['response'] = response

        # Store the current response for the next round if it's a follow up question.
        state["previous_response"] = response

        # Save to JSON chat history
        json_chat_history.add_interaction(
            question=state["question"],
            rephrased_question=state.get("rephrased_question"),
            response=response,
            category="sql",
            sql=state.get('sql'),
            sql_error=state.get('sql_error'),
            df=df_json  # Store the JSON string
        )

        return state

    except Exception as e:
        workflow_logger.error(f"Error parsing JSON or processing DataFrame: {e}")
        state['response'] = f"An error occurred while processing the data: {e}"
        json_chat_history.add_interaction(
            question=state["question"],
            rephrased_question=state.get("rephrased_question"),
            response=state['response'],
            category="sql",
            sql=state.get('sql'),
            sql_error=state.get('sql_error'),
            df=None
        )
        return state

def should_retry_sql(state):
    workflow_logger.info(f"should_retry_sql: sql_error={state.get('sql_error')}, sql_retries={state.get('sql_retries', 0)}")
    if state.get("sql_error") and state.get("sql_retries", 0) < max_retries:
        workflow_logger.info("Retrying SQL generation")
        return "sql_gen_node"
    workflow_logger.info("Moving to SQL response node")
    return "sql_response_node"

def gk_answer_node(state):
    # Enhanced instructions for general knowledge responses
    instructions = """You are a helpful AI assistant with broad knowledge across many topics.

    Guidelines for responses:
    1. For greetings and casual conversation, be friendly but concise
    2. For factual questions, provide accurate and direct answers
    3. For conceptual questions, explain clearly with simple examples
    4. Keep responses focused and relevant to the question
    5. If unsure, acknowledge limitations
    6. Maintain a professional but approachable tone

    Remember: You're handling non-database questions that don't require SQL queries.
    """

    # Get recent chat history for context
    chat_history = json_chat_history.get_recent_history(4)
    formatted_history = ""

    if chat_history:
        formatted_history = "Recent conversation history:\n"
        for i, entry in enumerate(chat_history):
            formatted_history += f"User: {entry.get('question', '')}\n"
            formatted_history += f"Assistant: {entry.get('response', '')}\n\n"

    # Check if there's a previous response
    previous_response = state.get("previous_response")

    # Include the previous response in the current prompt
    prompt = state.get("rephrased_question", state["question"])

    # Prepend the prompt with previous response
    if previous_response:
        prompt = f"{prompt}\n\nPrevious Response: {previous_response}"

    # Create the prompt string directly, embedding the history
    final_prompt = f"{instructions}\n{formatted_history}\n{prompt}"

    # Invoke the LLM with the prompt string
    messages = [HumanMessage(content=final_prompt)]
    chat_logger.info(f"Sending prompt to LLM: {final_prompt[:100]}...")
    response = llm.invoke(messages)
    state['response'] = response.content

    # Save to JSON chat history
    json_chat_history.add_interaction(
        question=state["question"],
        rephrased_question=state.get("rephrased_question"),
        response=response.content,
        category="gk"
    )

    return state

def preprocessor_node(state):
    
    class QuestionClassifier(BaseModel):
        category: Literal["gk", "sql"] = Field(
            ...,
            description="Classification of the question: 'gk' for general knowledge or 'sql' for SQL query required."
        )
        rephrased_question: str = Field(
            ...,
            description="A clear, concise, and standalone rephrasing of the user's question that incorporates the given chat history."
        )

    question = state["question"]
    chat_logger.info(f"preprocessor_node: User question = {question}")

    try:
        chat_history = json_chat_history.get_recent_history(4)
        formatted_history = ""
        for entry in chat_history:
            formatted_history += f"User: {entry['question']}\nAssistant: {entry['response']}\n\n"
    except (FileNotFoundError, json.JSONDecodeError):
        chat_history = []
        formatted_history = ""

    # Load dynamic follow-up question rules from a JSON file
    try:
        with open("follow_up_rules.json", "r") as f:
            follow_up_rules = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        chat_logger.warning("No follow-up rules loaded.")
        follow_up_rules = []

    for rule in follow_up_rules:
        keywords = rule.get("keywords", [])
        category = rule.get("category", "gk")
        rephrased_question_template = rule.get("rephrased_question_template", None)

        # Check if any keywords exist within the current question
        keyword_match = any(keyword in question for keyword in keywords)

        if keyword_match:
            #If any keyword exist, get the last entry
            if chat_history:
                last_entry = chat_history[-1]
                last_question = last_entry.get("question", "")

                #See if keywords match last question to fulfill second condition
                last_keyword_match = any(keyword in last_question for keyword in keywords)

                if last_keyword_match:
                    if rephrased_question_template:
                        # Format the rephrased question using the template and the last question
                        try:
                            rephrased_question = rephrased_question_template.format(
                                question=question, 
                                last_question=last_question
                            )

                            state["category"] = category
                            state["rephrased_question"] = rephrased_question

                            chat_logger.info(f"Follow-up detected, rephrasing to: {rephrased_question}, Category: {category}")
                            return state
                        except KeyError as e:
                            chat_logger.error(f"Error in rephrased question template: {e}")
                    else:
                        chat_logger.warning("No rephrased question template specified.")

                else:
                    chat_logger.info("No keywords found in previous question")
            else:
                chat_logger.info("No chat history available")

    # If not a follow-up, use dynamic prompt engineering
    INSTRUCTIONS = """
        You are a specialized assistant tasked with understanding user intent and preparing questions for a knowledge base.
        Rephrase the user's question into a clear, concise, and standalone version that incorporates the chat history. Pay very close attention to whether this is a follow-up question.  If it is, you **MUST** incorporate the previous question and its result into the rephrased question.
        Classify the question as either 'gk' (general knowledge) or 'sql' (requires database query).

    **Classification:** Classify the (potentially rephrased) question as either 'gk' (general knowledge) or 'sql' (requires database query).

        Examples:
           - "Hello, how are you?" -> Category: 'gk'
           - "What is SQL?" -> Category: 'gk'
           - "How many employees are in the Sales department?" -> Category: 'sql'
           - "Show me customer orders from last month" -> Category: 'sql'
           - "count all the products" -> Category: 'sql'

      ** If Classification is 'sql': ** Check the chat_history and rephrase the question to include the context of the conversation.
        EXAMPLES:
            - (Previous: "count all the products"), "What are the top 5?" -> Category: 'sql', Rephrased: "What are the top 5 products?"

        Your response must be in JSON format with exactly two keys: 'category' and 'rephrased_question'.
        **chat_history**: {chat_history}
    """

    # Load Few-Shot Examples
    FEW_SHOT_EXAMPLES = load_few_shot_examples()

    # Create Example Prompt Template
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    # Create Few-Shot Prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=FEW_SHOT_EXAMPLES,
        input_variables=["input"]
    )

    # Build Final Prompt Template
    messages = [
        ("system", INSTRUCTIONS),
        ("human", "{query}")
    ]

    # Optionally add few-shot examples if they exist
    if FEW_SHOT_EXAMPLES:
        messages.insert(1, few_shot_prompt)

    final_prompt = ChatPromptTemplate.from_messages(messages)

    # Invoke the LLM with the Prompt
    messages = final_prompt.invoke({
        "query": question,
        "chat_history": formatted_history,
        "input": question
    })

    structured_llm = llm.with_structured_output(QuestionClassifier)

    try:
        response = structured_llm.invoke(messages)
        state["category"] = response.category
        state["rephrased_question"] = response.rephrased_question
        chat_logger.info(f"Classification successful: Category={response.category}, Rephrased Question={response.rephrased_question}")
        return state
    except Exception as e:
        chat_logger.error(f"Error in classification: {e}")
        state["category"] = "gk"
        state["rephrased_question"] = question
        return state

def route(state):
    if state["category"] == "gk":
        return "gk_answer_node"
    else:
        return "sql_gen_node"

# Create workflow graph
workflow = StateGraph(dict)
workflow.add_node("preprocessor_node", preprocessor_node)
workflow.add_node("gk_answer_node", gk_answer_node)
workflow.add_node("sql_gen_node", sql_gen_node)
workflow.add_node("execute_sql_node", execute_sql_node)
workflow.add_node("sql_response_node", sql_response_node)

workflow.set_entry_point("preprocessor_node")
workflow.add_conditional_edges(
    "preprocessor_node",
    route,
    {
        "gk_answer_node": "gk_answer_node",
        "sql_gen_node": "sql_gen_node"
    }
)

workflow.add_edge("sql_gen_node", "execute_sql_node")
workflow.add_conditional_edges(
    "execute_sql_node",
    should_retry_sql,
    {
        "sql_gen_node": "sql_gen_node",
        "sql_response_node": "sql_response_node"
    }
)
workflow.add_edge("sql_response_node", END)
workflow.add_edge("gk_answer_node", END)

# Compile the graph
app_workflow = workflow.compile()

# Initialize Vanna with training data
def initialize_vanna():
    try:
        vn.load_model(path=settings.VANNA_MODEL_PATH)
        workflow_logger.info(f"Vanna model loaded from {settings.VANNA_MODEL_PATH}")
    except Exception as e:
        workflow_logger.warning(f"Vanna model not found at {settings.VANNA_MODEL_PATH}. Initializing a new model.")

        vn.load_ddl("""
        CREATE TABLE Customers (
            CustomerID INTEGER PRIMARY KEY,
            FirstName TEXT,
            LastName TEXT,
            City TEXT,
            Country TEXT
        );

        CREATE TABLE Orders (
            OrderID INTEGER PRIMARY KEY,
            CustomerID INTEGER,
            OrderDate TEXT,
            TotalAmount REAL,
            FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
        );
        """)

        vn.train(question="How many customers are there?", sql="SELECT COUNT(*) FROM Customers")
        vn.train(question="List all customer names", sql="SELECT FirstName, LastName FROM Customers")
        vn.train(question="What is the total order amount?", sql="SELECT SUM(TotalAmount) FROM Orders")

        vn.train(
            question="What is the average order amount for customers in France?",
            sql="""
            SELECT AVG(o.TotalAmount)
            FROM Orders o
            JOIN Customers c ON o.CustomerID = c.CustomerID
            WHERE c.Country = 'France'
            """,
            ddl = """
            CREATE TABLE Customers (
                CustomerID INTEGER PRIMARY KEY,
                FirstName TEXT,
                LastName TEXT,
                City TEXT,
                Country TEXT
            );

            CREATE TABLE Orders (
                OrderID INTEGER PRIMARY KEY,
                CustomerID INTEGER,
                OrderDate TEXT,
                TotalAmount REAL,
                FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
            );
            """
        )

        vn.save_model(path=settings.VANNA_MODEL_PATH)
        workflow_logger.info(f"Vanna model initialized and saved to {settings.VANNA_MODEL_PATH}")

# Initialize Vanna on startup (can be moved to a startup event)
try:
    initialize_vanna()
except Exception as e:
    workflow_logger.error(f"Error initializing Vanna: {e}")

@router.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, background_tasks: BackgroundTasks):
    """Process a chat message and return a response."""
    chat_logger.info(f"Received chat message: {message.question}")
    
    # Set up initial state
    state = {"question": message.question}

    try:
        graph_state = app_workflow.invoke(state)
        
        response_data = {
            "response": graph_state.get("response", "No response generated"),
            "category": graph_state.get("category", "unknown"),
            "rephrased_question": graph_state.get("rephrased_question"),
            "metadata": {
                "sql": graph_state.get("sql"),
                "sql_error": graph_state.get("sql_error"),
                "df": graph_state.get("df")
            }
        }
        
        chat_logger.info(f"Response generated for message: {message.question}")
        return response_data

    except Exception as e:
        chat_logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))