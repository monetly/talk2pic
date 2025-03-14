import torch
import os
from diffusers import FluxPipeline
from openai import OpenAI
import time

class flux:

    def __init__(self, seed, prompt=""):  
        self.seed = seed
        self.prompt = prompt
        self.pipe = None
        self.key="sk-scslpgwlfujtfgvphuwljnnstgbbokhdekkirgxgibfzgdss"

    def load_model(self, model_id,lora_id=None):
        if self.pipe is None:
            pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
            )
            if lora_id:
                pipe.load_lora_weights(lora_id)
                pipe.fuse_lora()
                pipe.enable_sequential_cpu_offload()
                print(f"Model Lora has been loaded: {lora_id}")
            self.pipe = pipe
            print(f"Model has been loaded: {model_id}")
        else:
            print(f"Model has been loaded: {model_id}")

    def generated_prompt(self, seed):
        prompt = ""
        self.seed = str(seed)
        client = OpenAI(api_key=self.key, base_url="https://api.siliconflow.cn/v1")
        response = client.chat.completions.create(
            model='deepseek-ai/DeepSeek-V3',
            messages=[  
                {"role": "system", "content": "You are now my assistant, and your task is to describe, in as much detail as possible, the potential scene of the dialogue I'm about to send you, using the following format: Character status, Action/Emotion, School Setting, Illustration Style, Color Palette. The scene description must be specific and accurate, avoiding vague terms and providing clear details. Answer in English. You don't need to send your response in the format provided; simply connect the content corresponding to each format and send it to me. Limit your response to 80 tokens."},  
                {"role": "user", "content": self.seed},  
            ],  
            stream=False
        )
        prompt=response.choices[0].message.content
        self.prompt=prompt

    def generated_pic(self, prompt=""):
        if prompt == "":
            prompt = self.prompt 
        image = self.pipe(
            prompt,
            negative_prompt="correct hand",
            height=720,
            width=1280,
            guidance_scale=3.5,
            num_inference_steps=20,
            max_sequence_length=512,
            generator=torch.manual_seed(0)
        ).images[0]
        print(prompt)
        image.save(f"./{self.seed[:4]}.webp")


    def batch_generated_pic(self,file):
        with open(file, "r", encoding="utf-8") as file:
            line = file.readline()
            while line:
                self.generated_prompt(line)
                self.generated_pic()
                time.sleep(0.2)
                line = file.readline()

    def release_model(self):
        if self.pipe is not None:
            del self.pipe 
            torch.cuda.empty_cache() 
            print("Model has been released and memory cleared.")
        else:
            print("No model to release.")
        

if __name__=='__main__':
    sd = flux("ruler")
    model_id="/home/dictmanage/.cache/modelscope/hub/models/tensorart/stable-diffusion-3.5-medium-turbo/"
    model_id1="/home/dictmanage/.cache/modelscope/hub/models/black-forest-labs/FLUX.1-dev/"
    adapter_id = "/home/dictmanage/.cache/modelscope/hub/models/yiwanji/FLUX_xiao_hong_shu_ji_zhi_zhen_shi_V2/"
    Child_Il="/home/dictmanage/.cache/modelscope/hub/models/lip421/ertongchahuaMAILANDFLUX/"
    sd.load_model(model_id1,Child_Il)
    sd.generated_pic('how are you my friend')
    sd.release_model()