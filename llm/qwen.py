from openai import OpenAI
from llm.utils import llm_utils as lm
import base64


def png_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/png;base64,{base64_str}"
    except FileNotFoundError:
        return None
    except Exception as e:
        return None



####QwenChatModel####
class qwen(lm):
    def __init__(self,key:str):
        super().__init__()
        self.key=key
        self.url="https://api.siliconflow.cn/v1"
        self.load_llm()
    def load_llm(self):
        self.client = OpenAI(api_key=self.key,base_url=self.url)
    def call_llm(self,seed:str):
        response = self.client.chat.completions.create(
            model='Qwen/QwQ-32B',
            messages=[
                {"role": "system", "content": "You are now my assistant, and your task is to describe, in as much detail as possible, the potential scene of the dialogue I'm about to send you, using the following format: Character status, Action/Emotion, Environment Setting, Illustration Style, Color Palette. The scene description must be specific and accurate, avoiding vague terms and providing clear details. Answer in English. You don't need to send your response in the format provided; simply connect the content corresponding to each format and send it to me. Limit your response to 80 tokens.Don't show the think/n and /space."},
                {"role": "user", "content": seed},
            ],
            stream=False
        )
        return response.choices[0].message.content





#####QwenVision#####
class vqwen(qwen):
    def __init__(self,key):
        super().__init__(key)
        self.image=None

    def set_image(self, image_path):
        self.image = png_to_base64(image_path)

    def load_llm(self):
        super().load_llm()
    
    def evaluate_image(self, prompt:str,image_path):
        self.set_image(image_path)
        response = self.client.chat.completions.create(
        model="Qwen/Qwen2-VL-72B-Instruct",
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self.image
                    }
                },
                {
                    "type": "text",
                    "text": "You are an AI that evaluates whether an image matches a given prompt. Compare the image with the provided prompt, and identify any inconsistencies such as incorrect hand positions, unnatural body poses, unrealistic lighting, missing objects,Gender-nonconforming clothing, or any significant mistakes. Give a score from 0 (perfectly accurate) to 10 (completely mismatched). Only return the score as an integer."
                }
            ]
        }],
        stream=False)
        try:
            score = int(response.choices[0].message.content.strip())
            print(f"Image evaluation score: {score}/10")
            return score
        except ValueError:
            print("Failed to get evaluation score. Assuming it's fine.")
            print(response.choices[0].message.content.strip())
            return 0

    def refine_prompt(self, old_prompt):

        response = self.client.chat.completions.create(
            model="Qwen/Qwen2-VL-72B-Instruct",
            messages=[
            {"role": "system", "content": [{"type": "text", "text": "Your task is to refine an image generation prompt to fix errors and improve accuracy. Given the prompt and a list of detected issues, generate an improved version of the prompt while keeping the original intent intact."}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": self.image}}, {"type": "text", "text": f"Original prompt: {old_prompt}\\nDetected issues: Incorrect hands, unnatural lighting, incorrect text, distorted perspective. Improve the prompt to avoid these issues."}]}
            ],
            stream=False
            )

        return response.choices[0].message.content.strip()

if __name__=="__main__":

    #b=vqwen("sk-qzvagazrsjcgrkqfmynotwtovonbqxcrfhmgpleutfovldgo")
    #b.evaluate_image("A BOY BARKS TO A DOG","/home/dictmanage/liuxh/talk2pic/image/animal/A do.webp")
    a=qwen("sk-qzvagazrsjcgrkqfmynotwtovonbqxcrfhmgpleutfovldgo")
    print(a.call_llm("hello,how are you").lstrip())

