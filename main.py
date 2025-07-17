from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from routes import chat_routes
import uvicorn

app = FastAPI(
    title="SQL Assistant API",
    description="An API for interacting with a SQL database through natural language",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# app.include_router(chat_routes.router, prefix="/api", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to SQL Assistant API. Navigate to /docs for documentation."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)