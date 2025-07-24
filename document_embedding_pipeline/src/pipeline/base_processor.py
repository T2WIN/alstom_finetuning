from abc import ABC, abstractmethod
from pathlib import Path

class BaseProcessor(ABC):
    
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    async def process(self, file_path: Path):
        pass