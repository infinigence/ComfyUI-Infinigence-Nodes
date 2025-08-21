import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
import requests
from sklearn.decomposition import PCA
# ------------------------------
# 字体缓存与工具
# ------------------------------
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_PATH=  os.path.join(FONT_DIR, "Arial-Unicode-Regular.ttf")
print("font dir:", FONT_DIR, "font path:", FONT_PATH)
_font_cache = {}

def ensure_font():
    if not os.path.exists(FONT_PATH):
        print(f"字体文件不存在, 路径：{FONT_PATH}")
        return None
    return FONT_PATH

# ------------------- 绘制文字函数 -------------------
def fit_font_size(text, mask_w, mask_h, font_path, max_size=200):
    for size in range(max_size, 5, -2):
        font = ImageFont.truetype(font_path, size)
        bbox = font.getbbox(text)
        text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        if text_w <= mask_w and text_h <= mask_h:
            return font, (text_w, text_h)
    return ImageFont.truetype(font_path, 12), (12, 12)

def draw_text_in_mask(base_img, mask, text, angle=0, color=(0,0,0)):
    """
    在 mask 区域中绘制文字，支持旋转
    """
    font_path = ensure_font()
    if font_path is None:
        return base_img
    mask_img = Image.fromarray((mask>0).astype(np.uint8)*255)
    bbox = mask_img.getbbox()
    if bbox is None:
        return base_img
    x0, y0, x1, y1 = bbox
    mask_w, mask_h = x1-x0, y1-y0

    font, (tw, th) = fit_font_size(text, mask_w, mask_h, font_path)
    text_layer = Image.new("RGBA", (mask_w, mask_h), (0,0,0,0))
    draw = ImageDraw.Draw(text_layer)
    draw.text(((mask_w-tw)//2,(mask_h-th)//2), text, font=font, fill=color+(255,))

    if angle != 0:
        # 关键修改：PCA角度是顺时针，PIL.rotate是逆时针
        angle_for_pil = -angle
        # 避免文字倒置，限制角度在 [-90,90]
        if angle_for_pil < -90:
            angle_for_pil += 180
        elif angle_for_pil > 90:
            angle_for_pil -= 180
        text_layer = text_layer.rotate(angle_for_pil, expand=True, resample=Image.BICUBIC)

    tw, th = text_layer.size
    paste_x = x0 + (mask_w - tw)//2
    paste_y = y0 + (mask_h - th)//2
    base_img.paste(text_layer, (paste_x, paste_y), text_layer)
    return base_img

# ------------------- PCA 计算主方向 -------------------
def get_contour_angle(cnt):
    pts = cnt.reshape(-1,2)
    pca = PCA(n_components=2)
    pca.fit(pts)
    # 主方向向量
    vec = pca.components_[0]
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    # 保证角度在 0~180
    if angle < 0: angle += 180
    return angle

# ------------------- 多区域渲染 -------------------
def render_glyph_multi(original, computed_mask, texts, auto_angle=True):
    mask_np = np.array(computed_mask.convert("L"))
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h<50: 
            continue
        regions.append((x,y,w,h,cnt))
    regions = sorted(regions, key=lambda r:(r[1]//10,r[0]))

    render_img = original.convert("RGBA")
    for i, region in enumerate(regions):
        if i >= len(texts): break
        text = texts[i].strip()
        if not text: continue
        cnt = region[4]

        # 计算主方向
        angle = 0
        if auto_angle:
            angle = get_contour_angle(cnt)

        mask_region = np.zeros(mask_np.shape, dtype=np.uint8)
        cv2.drawContours(mask_region,[cnt],-1,255,-1)
        render_img = draw_text_in_mask(render_img, mask_region, text, angle=angle, color=(0,0,0))
    return render_img.convert("RGB")

# ------------------- ComfyUI 节点 -------------------
class DrawTextNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image":("IMAGE",), "mask":("IMAGE",), "words":("STRING",{"multiline":True})}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "draw"
    CATEGORY = "Image/Text"

    def comfy2pil(self,image):
        arr = (image[0].cpu().numpy()*255).astype(np.uint8)
        if arr.shape[-1]==4: arr=arr[...,:3]
        return Image.fromarray(arr)

    def pil2comfy(self,pil):
        arr = np.array(pil).astype(np.float32)/255.0
        if arr.ndim==2: arr=np.stack([arr]*3,-1)
        elif arr.shape[-1]==4: arr=arr[...,:3]
        return torch.from_numpy(arr)[None,...]

    def draw(self,image,mask,words):
        original = self.comfy2pil(image)
        computed_mask = self.comfy2pil(mask)
        texts = [line.strip() for line in words.splitlines() if line.strip()]
        out_img = render_glyph_multi(original, computed_mask, texts, auto_angle=True)
        return (self.pil2comfy(out_img),)



