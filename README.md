
# 🧠 Natural Language to SQL Chatbot

This project is a **Natural Language to SQL Assistant**, enabling users to ask questions in plain English and receive answers based on SQL queries executed on an underlying database. Built using **FastAPI**, it integrates tools like **Vanna-like SQL generation**, **few-shot learning**, and **custom context memory** to assist with intelligent, conversational data access.

---

## 🚀 Features

- ✅ **Natural Language to SQL Conversion**  
  Translates English questions into executable SQL queries using a Vanna-inspired engine.

- 🧠 **Few-Shot Prompting**  
  Enhances query generation through pre-defined examples (few-shot learning).

- 💬 **Chat History Management**  
  Maintains chat context using a custom history handler to preserve continuity.

- 🔌 **Modular Architecture**  
  Clean separation of configuration, schemas, routing, and utilities.

- 🛠 **SQLite Backend**  
  Interacts with a SQLite database (`southco.db`) to run and fetch real-time query results.

---

## 🗂️ Project Structure

```
new/
│
├── main.py                    # Entry point for FastAPI app
├── routes/
│   └── chat_routes.py         # Chat-related API endpoints
├── schema/
│   └── chat_schemas.py        # Request/Response schemas
├── utills/
│   ├── vanna_extension.py     # Vanna-style SQL conversion logic
│   ├── few_shot.py            # Few-shot prompt management
│   └── chat_history.py        # Conversation memory
├── config/
│   ├── database.py            # Database connection config
│   ├── log_config.py          # Logging settings
│   └── settings.py            # Environment variables and paths
├── logs/
│   └── logger.py              # Custom logging
├── chroma.sqlite3             # Vector store (e.g., ChromaDB)
├── southco.db                 # Main business database (used for SQL execution)
└── test.ipynb                 # Jupyter notebook for dev/testing
```

---

## 🧑‍💻 How It Works

1. **User sends a question** via the `/chat` endpoint.
2. The system:
   - Uses `vanna_extension.py` to generate a SQL query.
   - Fetches relevant examples via `few_shot.py`.
   - Maintains conversation history with `chat_history.py`.
3. The SQL is executed against the SQLite database.
4. Results are returned in a clean, readable format.

---

## 📦 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone <repo_url>
   cd new
   ```

2. **Install dependencies with Poetry**
   ```bash
   poetry install
   ```

3. **Run the server**
   ```bash
   poetry run python main.py
   ```

---

## 📬 API Endpoint

### `POST /chat`

**Request Body:**
```json
{
  "query": "Show me the total revenue from last quarter."
}
```

**Response:**
```json
{
  "query": "SELECT SUM(revenue) FROM sales WHERE quarter = 'Q2';",
  "result": [["1200000"]]
}
```

---

## 📚 Technologies Used

- **FastAPI** – Web framework
- **Pydantic** – Data validation
- **SQLite** – Lightweight database
- **Vanna-style Model** – SQL generation from plain English
- **Few-Shot Prompting** – Enhanced language understanding
- **Poetry** – Dependency management

---

## 📝 Future Improvements

- [ ] Add authentication and user management
- [ ] Expand support for other databases (PostgreSQL, MySQL)
- [ ] Integrate vector search from `chroma.sqlite3`
- [ ] Frontend UI with chat interface

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
