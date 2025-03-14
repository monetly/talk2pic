import torch
import os
from diffusers import StableDiffusion3Pipeline
from openai import OpenAI
import time

class sd:

    def __init__(self, seed, prompt=""):  
        self.seed = seed
        self.prompt = prompt
        self.pipe = None
        self.key="sk-scslpgwlfujtfgvphuwljnnstgbbokhdekkirgxgibfzgdss"

    def load_model(self, model_id):
        if self.pipe is None:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            self.pipe = self.pipe.to('cuda')
        else:
            print(f"Model has been loaded: {model_id}")

    def generated_prompt(self, seed):
        prompt = ""
        self.seed = seed
        client = OpenAI(api_key=self.key, base_url="https://api.siliconflow.cn/v1")
        response = client.chat.completions.create(
            model='deepseek-ai/DeepSeek-V3',
            messages=[  
                {"role": "system", "content": "You are an illustrator for elementary school textbooks, and your task is to train new illustrators. Use your imagination to describe the English words accuracy.I provide in a way that other illustrators can easily understand, helping young students remember the words. Your output should not be divided into paragraphs, and you should not include any additional content—go straight to the point.Limit in 77 Tokens."},  
                {"role": "user", "content": self.seed},  
            ],  
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                self.prompt += chunk.choices[0].delta.content

    def generated_pic(self, prompt=""):
        os.makedirs(f"./{self.seed}", exist_ok=True)
        if prompt == "":
            prompt = self.prompt 
        for i in range(5):
            image = self.pipe(
                prompt,
                num_inference_steps=22,
                guidance_scale=1.5,
                height=640,
                width=640
            ).images[0]
            image.save(f"./{self.seed}/{self.seed}{i}.webp")

    def batch_generated_pic(self,file):
        with open(file, "r", encoding="utf-8") as file:
            line = file.readline()
            while line:
                self.generated_prompt(line)
                self.generated_pic()
                time.sleep(0.5)
                line = file.readline

    def release_model(self):
        if self.pipe is not None:
            del self.pipe 
            torch.cuda.empty_cache() 
            print("Model has been released and memory cleared.")
        else:
            print("No model to release.")