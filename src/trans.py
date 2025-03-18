import torch
import os
from openai import OpenAI
from llm.qwen import qwen,vqwen
from llm.deepseek import deepseek,vdeepseek
import os
from PIL import Image
import time
from flux import flux

class trans:
    def __init__(self, api_key, output_dir="output_images"):
        self.key = api_key
        self.base_url = "https://api.siliconflow.cn/v1"
        self.pipe = None
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            self.llm=qwen(api_key)
        except:
            self.llm=deepseek(api_key)
        try:
            self.vlm=vqwen(api_key)
        except:
            self.vlm=vdeepseek(api_key)
        self.pipe=flux()

    def generated_prompt(self, seed):
        return self.llm.call_llm(seed)

    def generate_image(self, prompt, index):
        if self.pipe is None:
            print("Error: FLUX Not Loaded!")
            return None

        attempt = 0
        while attempt < 5:
            print(f"Generating image (Attempt {attempt + 1})...")
            timestamp = int(time.time())
            filename = os.path.join(self.output_dir, f"image_{index:03d}_attempt{attempt}_{timestamp}.png")
            image=self.pipe.generate_image(prompt,index,attempt,filename)
            image.save(filename)
            print(f"Saved: {filename}")

            score = self.vlm.evaluate_image(prompt, filename)
            if score <= 6:
                print(f"✅ Image is acceptable with score {score}/10.")
                return filename
            else:
                print(f"❌ Too many issues detected (score {score}/10). Refining prompt...")
                prompt = self.vlm.refine_prompt(prompt)

            attempt += 1

        print("⚠️ Maximum attempts reached. Using last generated image.")
        return filename

    def batch_generate(self, text_file):
        with open(text_file, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        print(f"Processing {len(lines)} dialogues...")

        for idx, line in enumerate(lines):
            prompt = self.generated_prompt(line)
            self.generate_image(prompt, idx)

if __name__=="__main__":
    key="sk-qzvagazrsjcgrkqfmynotwtovonbqxcrfhmgpleutfovldgo"
    a=trans(key)
    a.batch_generate("/home/dictmanage/liuxh/talk2pic/tests/park/park.txt")
