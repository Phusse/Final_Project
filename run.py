import uvicorn

if __name__ == "__main__":
    # This is equivalent to running 'uvicorn app.main:app --reload' in the terminal
    # reload=True enables auto-restart when you change code (good for development)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)