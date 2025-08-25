import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    #debugpy.listen(("localhost", 9503))
    #print("Waiting for debugger attach")
    #debugpy.wait_for_client()
    pass
except Exception as e:
    pass

import base64
import requests
import os
from io import BytesIO
import numpy as np
from PIL import Image
import torch

def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


class Qwen2VL_api:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "url": ("STRING", {"default": "",}),
                "api_key": ("STRING", {"default": "",}),
                "model": (
                    [
                        "qwen2.5-vl-72b-instruct",
                    ],
                    {"default": "qwen2.5-vl-72b-instruct"},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        text,
        url,
        api_key,
        model,
        temperature,
        image,
    ):
        #torch.save(image, '11111.pt')
        image_pil = tensor_to_pil(image)
        buffer = BytesIO()
        image_pil.save(buffer, format="PNG")  # 或 "PNG" 根据需要      
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": text,
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "stream": False
        }
       
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()['choices'][0]['message']['content']
        return result


if __name__ == '__main__':

    temp = Qwen2VL_api()
    text = """
    请基于图像内容和文字信息，生成一段用于图生视频（Image-to-Video）的高质量 prompt，要求：
    不返回识别的文字本身；
    将文字所表达的动作自然融入场景描述中，需要结合文本的位置，例如将左侧文本的“一个机器人向右行走”转化为“一个机器人正在画面左侧向右移动”；
    语言流畅，适合输入至视频生成模型（如 Runway、Pika、Stable Video 等），能引导模型生成连贯、合理的动画效果。
    请输出最终的视频生成 prompt，不要解释过程。 
    """
    url = "https://cloud.infini-ai.com/maas/v1/chat/completions"
    api_key = "sk-c64iuwx6ijzop2ru"
    model =  "qwen2.5-vl-72b-instruct"
    temperature = 0.7
    image = torch.load('11111.pt')
    result = inference(
                text,
                url,
                api_key,
                model,
                temperature,
                image,
            )
    print(result)
