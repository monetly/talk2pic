from abc import ABC, abstractmethod

class llm_utils(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def load_llm(self,key:str):
        pass
    
    @abstractmethod
    def call_llm(self,seed:str):
        pass