import uvicorn
from src.config import settings

if __name__ == "__main__":
    uvicorn.run("src.server:app", host=settings.HOST, port=settings.PORT, reload=True)
