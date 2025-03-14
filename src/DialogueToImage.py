import torch
import os
from diffusers import FluxPipeline
from openai import OpenAI
from PIL import Image

import os
import time

class DialogueToImage:
    def __init__(self, api_key, model_id, lora_id=None, output_dir="output_images"):
        self.key = api_key
        self.base_url = "https://api.siliconflow.cn/v1"
        self.pipe = None
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_flux_model(model_id, lora_id)

    def load_flux_model(self, model_id, lora_id=None):
        if self.pipe is None:
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16
            )
            if lora_id:
                pipe.load_lora_weights(lora_id)
                pipe.fuse_lora()
                pipe.enable_sequential_cpu_offload()
                print(f"Model LoRA has been loaded: {lora_id}")
            self.pipe = pipe
            print(f"Model has been loaded: {model_id}")
        else:
            print(f"Model already loaded: {model_id}")

    def generated_prompt(self, seed):
        client = OpenAI(api_key=self.key, base_url=self.base_url)
        response = client.chat.completions.create(
            model='deepseek-ai/DeepSeek-V3',
            messages=[
                {"role": "system", "content": "You are now my assistant, and your task is to describe, in as much detail as possible, the potential scene of the dialogue I'm about to send you, using the following format: Character status, Action/Emotion, School Setting, Illustration Style, Color Palette. The scene description must be specific and accurate, avoiding vague terms and providing clear details. Answer in English. You don't need to send your response in the format provided; simply connect the content corresponding to each format and send it to me. Limit your response to 80 tokens."},
                {"role": "user", "content": seed},
            ],
            stream=False
        )
        return response.choices[0].message.content

    def generate_image(self, prompt, index):
        if self.pipe is None:
            print("Error: FLUX 模型未加载！")
            return None

        attempt = 0
        while attempt < 5:
            print(f"Generating image (Attempt {attempt + 1})...")

            image = self.pipe(
                prompt=prompt,
                negative_prompt="correct hand",
                height=720,
                width=720,
                guidance_scale=3.5,
                num_inference_steps=12,
                max_sequence_length=512,
                generator=torch.manual_seed(index + attempt)
            ).images[0]

            # 使用时间戳生成唯一的文件名
            timestamp = int(time.time())
            filename = os.path.join(self.output_dir, f"image_{index:03d}_attempt{attempt}_{timestamp}.png")
            image.save(filename)
            print(f"Saved: {filename}")

            score = self.evaluate_image(prompt, filename)
            if score <= 6:
                print(f"✅ Image is acceptable with score {score}/10.")
                return filename
            else:
                print(f"❌ Too many issues detected (score {score}/10). Refining prompt...")
                prompt = self.refine_prompt(prompt)

            attempt += 1

        print("⚠️ Maximum attempts reached. Using last generated image.")
        return filename  # 如果尝试 5 次仍不合理，返回最后一次生成的图片

    def evaluate_image(self, prompt, image_path):
        """使用 LLM 评估图片合理性"""
        client = OpenAI(api_key=self.key, base_url=self.base_url)
        image = Image.open(image_path)

        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {"role": "system", "content": "You are an AI that evaluates whether an image matches a given prompt. Compare the image with the provided prompt, and identify any inconsistencies such as incorrect hand positions, unnatural body poses, unrealistic lighting, missing objects, or any significant mistakes. Give a score from 0 (perfectly accurate) to 10 (completely mismatched). Only return the score as an integer."},
                {"role": "user", "content": f"Prompt: {prompt}\nImage: {image_path}"}
            ],
            stream=False
        )

        try:
            score = int(response.choices[0].message.content.strip())
            print(f"Image evaluation score: {score}/10")
            return score
        except ValueError:
            print("Failed to get evaluation score. Assuming it's fine.")
            return 0

    def refine_prompt(self, old_prompt):
        client = OpenAI(api_key=self.key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {"role": "system", "content": "Your task is to refine an image generation prompt to fix errors and improve accuracy. Given the prompt and a list of detected issues, generate an improved version of the prompt while keeping the original intent intact."},
                {"role": "user", "content": f"Original prompt: {old_prompt}\nDetected issues: Incorrect hands, unnatural lighting, distorted perspective. Improve the prompt to avoid these issues."}
            ],
            stream=False
        )

        return response.choices[0].message.content.strip()

    def batch_generate_from_file(self, text_file):
        with open(text_file, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        print(f"Processing {len(lines)} dialogues...")

        for idx, line in enumerate(lines):
            prompt = self.generated_prompt(line)
            self.generate_image(prompt, idx)


