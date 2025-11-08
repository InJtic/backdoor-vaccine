# utils/hf.py
import huggingface_hub as hf
from dotenv import load_dotenv
import os


def login():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    hf.login(token=hf_token)
