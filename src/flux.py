from diffusers import FluxPipeline
import torch
import time
class flux:
    def __init__(self,model_id="/home/dictmanage/.cache/modelscope/hub/models/black-forest-labs/FLUX.1-dev/", lora_id="/home/dictmanage/.cache/modelscope/hub/models/lip421/ertongchahuaMAILANDFLUX/"):
        self.pipe=None
        self.load_model(model_id,lora_id)


    def load_model(self, model_id, lora_id=None):
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
    
    def generate_image(self, prompt, index,attempt,filename):
        if self.pipe is None:
            print("Error: FLUX Not Loaded!")
            return None
        image = self.pipe(
            prompt=prompt,
            negative_prompt="correct hand",
            height=960,
            width=960,
            guidance_scale=3.5,
            num_inference_steps=12,
            max_sequence_length=512,
            generator=torch.manual_seed(index + attempt)
            ).images[0]
        return image
            

if __name__=="__main__":
    a=flux()
    a.generate_image("aksjjksa",1)