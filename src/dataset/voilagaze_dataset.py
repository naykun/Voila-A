import base64
from io import BytesIO
import re
import contextlib
import os
import orjson
import ijson.backends.yajl2_c as ijson
from PIL import ImageFile, Image
from torchvision import transforms
import random
import cv2
import sys
import torch
import numpy as np
sys.path.append(".")
# from .transforms import *

# from transforms import *


from torch.utils.data import Dataset


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

FLAMINGO_MEAN = [0.481, 0.458, 0.408]
FLAMINGO_STD = [0.269, 0.261, 0.276]

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

class VoilaDataset(Dataset):
    def __init__(
        self,
        args,
        cur_voila_path,
        cur_images_path,
        cur_train_config_path,
        is_test=False,
        supported_data_types=["caption", "qa"],
    ):
        self.args = args
        self.task_name = args.task
        self.is_test = is_test
        self.tokenizer = args.tokenizer

        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length

        self.seed = args.seed
        self.patch_image_size = args.patch_image_size
        self.supported_data_types = supported_data_types

        self.epoch = 0

        scales = [(args.patch_image_size, args.patch_image_size)]

        self.patch_resize_transform = transforms.Compose(
            [
                transforms.Resize((args.patch_image_size, args.patch_image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
            ]
        )

        self.voila_path = cur_voila_path
        self.images_path = cur_images_path
        self.train_config_path = cur_train_config_path

        assert os.path.exists(cur_voila_path), f"Error: The local voila_path {cur_voila_path} not exists!"

        assert os.path.exists(cur_images_path), f"Error: The local images_path {cur_images_path} not exists!"

        assert os.path.exists(cur_train_config_path), f"Error: The local train_config_path {cur_train_config_path} not exists!"

        # Load the dataset
        with open(self.voila_path, "rb") as f:
            self.dataset = orjson.loads(f.read())

        # Load the images
        with open(self.images_path, "rb") as f:
            self.images = orjson.loads(f.read())

        # Load the train_config
        with open(self.train_config_path, "rb") as f:
            self.train_config = orjson.loads(f.read())

        self.train_data_list = list(set(self.train_config.keys()) & set(self.dataset.keys()))

        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

    def random_init_case(self, question):
        if len(question) == 0:
            return question

        first_letter = question[0]
        if random.choice([True, False]):
            first_letter = first_letter.upper()
        else:
            first_letter = first_letter.lower()

        return first_letter + question[1:]

    def pre_question(self, question, max_ques_words):
        question = question.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ")
        question = self.random_init_case(question)

        question = re.sub(
            r"\s{2,}",
            " ",
            question,
        )
        question = question.lstrip("\n")
        question = question.rstrip("\n")
        question = question.strip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > max_ques_words:
            question = " ".join(question_words[:max_ques_words])

        return question

    def pre_answer(self, answer, max_ans_words):
        answer = re.sub(
            r"\s{2,}",
            " ",
            answer,
        )
        answer = answer.rstrip("\n")
        answer = answer.strip(" ")

        # truncate question
        return_answer = ""
        answers = answer.split(".")

        for _ in answers:
            if return_answer == "":
                cur_answer = _
            else:
                cur_answer = ".".join([return_answer, _])
            if len(cur_answer.split(" ")) <= max_ans_words:
                return_answer = cur_answer
            else:
                break

        if return_answer == "":
            answer_words = answer.split(" ")
            return_answer = " ".join(answer_words[:max_ans_words])
        else:
            if return_answer[-1] != "." and return_answer != answers:
                return_answer += "."

        return return_answer

    def pre_caption(self, caption, max_words):
        caption = caption.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ").replace("<person>", "person")

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > max_words:
            caption = " ".join(caption_words[:max_words])

        return caption

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def resample_frames(self, image_ids, resample_frames):
        indices = np.linspace(0, len(image_ids) - 1, resample_frames, dtype=int)
        image_ids = [image_ids[i] for i in indices]
        assert len(image_ids) == resample_frames
        return image_ids
    
    def get_heatmap(self, gaze):
        gazeCanvas = np.zeros((224, 224), dtype=np.float64)
        for p in gaze:
            x, y = min(max(p[0], 0.0), 0.999), min(max(p[1], 0.0), 0.999)
            x, y = int(x * gazeCanvas.shape[1]), int(y * gazeCanvas.shape[0])
            gazeCanvas[y, x] = 1.0
        gazeCanvas = cv2.GaussianBlur(gazeCanvas, (199, 199), 0)
        gazeCanvas = (gazeCanvas - np.min(gazeCanvas))
        gazeCanvas /= np.max(gazeCanvas)
        return gazeCanvas
    
    def process_trace_data(self, trace_data, target_shape=(40, 2)):
        trace_data = np.array(trace_data)
        if trace_data.shape[0] == 0:
            raise ValueError("The trace data is empty!")
        # if trace_data.shape[0] == target_shape[0]:
        #     return torch.tensor(trace_data, dtype=torch.float32)
        
        # Resample the trace data if the size is larger than the target shape
        if trace_data.shape[0] > target_shape[0]:
            step = trace_data.shape[0] / target_shape[0]
            indices = np.arange(0, trace_data.shape[0], step)
            indices = np.round(indices).astype(int)
            resampled_data = trace_data[indices]
            mask = torch.ones(target_shape[0], dtype=torch.int64)
        # Pad the trace data if the size is smaller than the target shape
        else:
            padding = target_shape[0] - trace_data.shape[0]
            # padded_data = np.pad(trace_data, ((0, padding), (0, 0)), mode='edge')
            trace_data = torch.tensor(trace_data, dtype=torch.float32)
            padded_data = torch.cat([trace_data, torch.zeros((padding, 2))], dim=0)
            resampled_data = padded_data
            mask = torch.cat([torch.ones(trace_data.shape[0], dtype=torch.int64),
                          torch.zeros(padding, dtype=torch.int64)], dim=0)
        
        return torch.tensor(resampled_data, dtype=torch.float32), mask

    def trace_to_token(self, trace_data, grid_size=(16, 16)):
        tensor_data, mask = self.process_trace_data(trace_data)
        tensor_data = (tensor_data * (grid_size[0] - 1)).round().long()
        trace_token = tensor_data[:, 0] * grid_size[1] + tensor_data[:, 1]
        trace_token = torch.clamp(trace_token, 0, grid_size[0] * grid_size[1] - 1)
        return trace_token, mask
        

    def process_voila(self, instruction_id, instruction, indirect_instruction, answer, trace, image_ids, in_context_exmaple_ids, inst_format="simple"):
        patch_images = torch.tensor([])
        all_texts = ""
        wrap_sys = f"<<SYS>>\nYou are a helpful vision language assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n"
        all_instruction_ids = in_context_exmaple_ids + [instruction_id]
        all_instruction_ids = [(idx, False) for idx in all_instruction_ids[:2]]
        all_instruction_ids[-1] = (all_instruction_ids[-1][0], True)
        for idx, (cur_instruction_id, use_gaze) in enumerate(all_instruction_ids):
            cur_instruction_image_id = self.dataset[cur_instruction_id]["image_ids"][0]
            cur_instruction = self.dataset[cur_instruction_id]["question"]
            cur_indirect_instruction = self.dataset[cur_instruction_id]["indirect question"]
            cur_trace = self.dataset[cur_instruction_id]["trace"]
            cur_answer = self.dataset[cur_instruction_id]["answer"]
            cur_instruction = self.pre_question(cur_instruction, self.max_src_length)
            cur_indirect_instruction = self.pre_question(cur_indirect_instruction, self.max_src_length)
            cur_answer = self.pre_answer(cur_answer, self.max_tgt_length)
            if use_gaze:
                cur_instruction=f"<fixation>{cur_indirect_instruction}"
            if inst_format == "llama2":
                if idx == 0:
                    cur_text = f"[INST]{wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
            elif inst_format == "idefics":
                if idx == 0:
                    cur_text = (
                        f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction} Assistant:<answer>{cur_answer}<|endofchunk|>"
                    )
                else:
                    cur_text = f"User:{cur_instruction} Assistant:<answer>{cur_answer}<|endofchunk|>"
            elif inst_format == "simple":
                if idx == 0:
                    cur_text = f"<image>User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"User:{cur_instruction} GPT:<answer>{cur_answer}<|endofchunk|>"
            all_texts += cur_text
        cur_image_id = self.dataset[instruction_id]["image_ids"][0]
        cur_image = self.images["VL_IMG_{:012d}".format(int(cur_image_id))]
        cur_image = Image.open(BytesIO(base64.urlsafe_b64decode(cur_image))).convert("RGB")
        patch_images = self.patch_resize_transform(cur_image).unsqueeze(0).unsqueeze(0)
        heatmap = self.get_heatmap(self.dataset[instruction_id]["trace"])
        heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
        trace_tokens, trace_mask = self.trace_to_token(self.dataset[instruction_id]["trace"])
        trace_data, _ = self.process_trace_data(self.dataset[instruction_id]["trace"])
        trace_box = [trace_data[:, 0].min(), trace_data[:, 1].min(), trace_data[:, 0].max(), trace_data[:, 1].max()]
        trace_box = trace_box + [(trace_box[2] - trace_box[0])*(trace_box[3] - trace_box[1])]
        trace_box = torch.tensor(trace_box)
        trace_box = torch.clamp(trace_box,0,1)
        
        return patch_images, all_texts, heatmap, trace_tokens, trace_mask, trace_box
    
    def __getitem__(self, index):
        cur_train_id = self.train_data_list[index]  
        (
            instruction_id,  
            instruction,  
            indirect_instruction,  
            answer, 
            trace,
            image_ids,  
            in_context_example_ids,  
        ) = (
            cur_train_id,
            self.dataset[cur_train_id]["question"],
            self.dataset[cur_train_id]["indirect question"],  
            self.dataset[cur_train_id]["answer"],
            self.dataset[cur_train_id]["trace"],
            self.dataset[cur_train_id]["image_ids"],
            self.train_config[cur_train_id],
        )

        self.max_src_length = self.max_tgt_length = 256

        patch_images, all_texts, trace_heatmap, trace_tokens, trace_mask, trace_box = self.process_voila(instruction_id, instruction, indirect_instruction, answer, trace, image_ids, in_context_example_ids)
        

        src_text = self.tokenizer(
            f"{all_texts}",
            return_tensors="pt",
            add_special_tokens=False,
        )

        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])  # TODO：bos、eos怎么使用
        src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])
        # src_item = torch.cat([self.bos_item, src_item])
        # src_item_mask = torch.cat([self.bos_mask, src_item_mask])

        example = {
            "id": instruction_id,
            "source": src_item,
            "text_mask": src_item_mask,
            "patch_images": patch_images,
            "trace_heatmap": trace_heatmap,
            "trace_tokens": trace_tokens, 
            "trace_mask": trace_mask,
            "trace_box": trace_box,
        }

        return example

    def __str__(self):
        return f"type: {type(self)}, length: {len(self)}"

    def __len__(self):
        return len(self.train_data_list)




if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    import json
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="voila")
    parser.add_argument("--max_src_length", type=int, default=256)
    parser.add_argument("--max_tgt_length", type=int, default=256)
    parser.add_argument("--patch_image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--data_dir", type=str, default="/mnt/lustre/yhzhang/Otter/pipeline/multi_instruct_data_utils/data")
    parser.add_argument("--voila_path", type=str, default="voila_anno.json")
    parser.add_argument("--images_path", type=str, default="voila_image.json")
    parser.add_argument("--train_config_path", type=str, default="voila_meta.json")
    args = parser.parse_args()

    args.tokenizer = text_tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b-instruct")
    dataset = VoilaDataset(args, args.voila_path, args.images_path, args.train_config_path)
    num = len(dataset)
    for i in tqdm(range(num)):
        try:
            data = dataset[i]
            print(data)
            print(data.keys())
        except Exception as e:
            raise e
        break