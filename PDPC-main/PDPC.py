import spacy
import torch
import clip
from PIL import Image
import torchvision.transforms as transforms
from itertools import product
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import Blip2ForConditionalGeneration, Blip2Processor
import warnings
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


#LLAMA3
llama = AutoModelForCausalLM.from_pretrained(
        "LLM-Research/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.float32,
        device_map="auto"
    )
llama_tokenizer = AutoTokenizer.from_pretrained("LLM-Research/Meta-Llama-3-8B-Instruct")

# load spaCy
nlp = spacy.load("en_core_web_sm")
device = "cpu"

#load CLIP
clip_model, preprocess = clip.load("ViT-L/14", device=device)

# load baseline
blip_processor = Blip2Processor.from_pretrained("blip2-opt-flan-t5-x")
blip_model = Blip2ForConditionalGeneration.from_pretrained("blip2-opt-flan-t5-x")

def get_dependency_pairs(text):
    doc = nlp(text)
    pairs = []
    for token in doc:
        if token.pos_ == "NOUN":  
            for child in token.children: 
                if child.dep_ in ("amod", "det", "prep", "pobj", "case", "compound"):
                    pairs.append((child.text, token.text, child.dep_))
    return pairs

def get_amod_dependencies(pairs, noun):
    return [dep for dep in pairs if dep[1] == noun and dep[2] == "amod"]

def get_compound_dependencies(pairs, noun):
    return [dep for dep in pairs if dep[1] == noun and dep[2] == "compound"]

def calculate_clip_score(image, text):
    image_inputs = preprocess(image).unsqueeze(0)
    text_inputs = clip.tokenize(text)
    logit_img_text, logit_text = clip_model(image_inputs, text_inputs)
    return logit_img_text


def generate_concept(image):
    annotations_path = "support_data.json"
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    device = "cpu"

    image_inputs = preprocess(image).unsqueeze(0).to(device)

    captions = [ann["caption"] for ann in coco_data["annotations"][:100]]

    text_inputs = clip.tokenize(captions).to(device)

    with torch.no_grad():
        logit_img_text, logit_text = clip_model(image_inputs, text_inputs)
        probs_text = logit_img_text.softmax(dim=-1).to('cpu').detach().numpy()

    k = 3
    top_k_indices = probs_text.argsort()[0, -k:][::-1]
    top_k_captions = [captions[i] for i in top_k_indices]

    concept = []
    for caption in top_k_captions:
        concept.append(caption)

    return concept
    

def replace_text(original_text, image):
    original_pairs = get_dependency_pairs(original_text)
    concept_text = generate_concept(image)
    all_concept_pairs = []

    for concept_text in concept_text:
        concept_pairs = get_dependency_pairs(concept_text)
        all_concept_pairs.extend(concept_pairs)

    original_nouns = set([pair[1] for pair in original_pairs if pair[2] in ["amod", "compound"]])
    concept_nouns = set([pair[1] for pair in all_concept_pairs if pair[2] in ["amod", "compound"]])

    best_combinations = []
    for noun in original_nouns.union(concept_nouns):
        ori_amods = get_amod_dependencies(original_pairs, noun)
        con_amods = get_amod_dependencies(all_concept_pairs, noun)
        ori_compounds = get_compound_dependencies(original_pairs, noun)
        con_compounds = get_compound_dependencies(all_concept_pairs, noun)

        best_score = -float('inf')
        best_combination = None

        all_amods = ori_amods + con_amods
        all_compounds = ori_compounds + con_compounds
        amod_groups = product(*[[amod] for amod in all_amods])
        compound_groups = product(*[[compound] for compound in all_compounds])

        for amod_group in amod_groups:
            for compound_group in compound_groups:
                combination_text = ' '.join([amod[0] for amod in amod_group] + [compound[0] for compound in compound_group] + [noun])
                logit_img_text = calculate_clip_score(image, combination_text)
                score = logit_img_text.softmax(dim=-1).detach().numpy()
                if score > best_score:
                    best_combination = amod_group + compound_group
                    best_score = score

        if best_combination:
            attributes = {
                "amod": {},
                "compound": {}
            }
            for word in best_combination:
                attr_type = word[2]  
                if word[1] not in attributes[attr_type]:
                    attributes[attr_type][word[1]] = word
                else:
                    existing_combination_text = ' '.join([attributes[attr_type][word[1]][0], word[1]])
                    candidate_combination_text = ' '.join([word[0], word[1]])
                    existing_score = calculate_clip_score(image, existing_combination_text).softmax(dim=-1).detach().numpy()
                    candidate_score = calculate_clip_score(image, candidate_combination_text).softmax(dim=-1).detach().numpy()
                    if candidate_score > existing_score:
                        attributes[attr_type][word[1]] = word

            best_combinations.append([(attributes["amod"].get(noun, (None, noun))[0], noun),
                                      (attributes["compound"].get(noun, (None, noun))[0], noun)])
            print("**&&**", best_combinations)
    
    if best_combination:
            best_combinations.append([(word[0], noun) for word in best_combination])

    best_combinations = [pair for sublist in best_combinations for pair in sublist]
    best = list(set(best_combinations))
    # print("best: ", best)

    merged_dict = {}
    for item in best:
        key = item[1]  
        if key in merged_dict:
            merged_dict[key].append(item[0])  
        else:
            merged_dict[key] = [item[0]]  

    merged_list = [(tuple(value) + (key,)) for key, value in merged_dict.items()]
    print(merged_list)

    rebuild_text = llama3(merged_list, original_text)

    return rebuild_text


def llama3(keyword_pairs, ori_text):
    device = "cuda" # the device to load the model onto
    
    warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set*", category=UserWarning)
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id*:", category=UserWarning)

    examples = [
        {"role": "user", "content": "Use these to revise the original sentence: a leaf with brown spots on it:[('middle', 'part'), ('brown', 'spots'), ('brown', 'leaf'), ('random', 'spot'), ('middle', 'area')]"},
        {"role": "assistant", "content": "A brown leaf with brown random spots in the middle area."},
        {"role": "user", "content": "[Use these to revise the original sentence: a leaf with brown spots on it:('left', 'side'), ('yellow', 'spots'), ('yellow', 'leaf'), ('scattered', 'spots'), ('left', 'area')]"},
        {"role": "assistant", "content": "A yellow leaf with scattered yellow spots on the left side."},
        {"role": "user", "content": "[Use these to revise the original sentence: a leaf with brown spots on it:('left', 'side'), ('black', 'spots'), ('green', 'leaf'), ('irregular', 'spots')]"},
        {"role": "assistant", "content": "A green leaf with irregular black spots on the left side."},
        {"role": "user", "content": "[Use these to revise the original sentence: a flower with brown spots on it:('left', 'side'), ('black', 'spots'), ('green', 'leaf'),('purple', 'leaf'), ('irregular', 'spots'),('yellow', 'spot'), ('large', 'area')]"},
        {"role": "assistant", "content": "A green leaf with irregular black spots on the left cover most of the area"},
        {"role": "user", "content": "The 'part' and 'area' attributes are used to describe spots"}
    ]

    examples.append({"role": "user", "content": "Use these to revise the original sentence:"+" "+ ori_text +" "+str(keyword_pairs)})

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        *examples
    ]
    text = llama_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = llama_tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = llama.generate(
        model_inputs.input_ids,
        max_new_tokens=50
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def PDPC(image):
    inputs = blip_processor(images=image, return_tensors="pt")  # .to(device, torch.float16)

    generated_ids = blip_model.generate(**inputs)
    original_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    response = replace_text(original_text, image)
    return response


# test 
image_path = "test.jpg"
image = Image.open(image_path)    
text = PDPC(image)
print(text)


