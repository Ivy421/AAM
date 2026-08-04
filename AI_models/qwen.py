import base64
import copy
import json
import mimetypes
import os

os.environ["DASHSCOPE_API_KEY"] = "sk-d7cc50dfb91b4f66b14bb03282057fd4"

def _image_to_data_url(image_path):
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def _convert_messages_for_qwen_api(messages):
    api_messages = copy.deepcopy(messages)

    for message in api_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        converted_content = []
        for item in content:
            if item.get("type") == "image":
                image_ref = item.get("image")
                if image_ref is None:
                    continue

                if str(image_ref).startswith(("http://", "https://", "data:image")):
                    image_url = str(image_ref)
                else:
                    image_url = _image_to_data_url(image_ref)

                converted_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })
            else:
                converted_content.append(item)

        message["content"] = converted_content

    return api_messages


def qwen3_inference(
    messages,
    model_name="qwen3-8b",
    max_tokens=256,
    temperature=0.0,
):
    """
    Qwen API inference through Alibaba Cloud Bailian/DashScope OpenAI-compatible API.

    Required environment variable:
        DASHSCOPE_API_KEY

    Optional environment variable:
        DASHSCOPE_BASE_URL
        default: https://dashscope.aliyuncs.com/compatible-mode/v1

    Keep the old return style: [model_output_text].
    """
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set.")

    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    api_messages = _convert_messages_for_qwen_api(messages)

    response = client.chat.completions.create(
        model=model_name,
        messages=api_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"enable_thinking": False},
    )

    return [response.choices[0].message.content]


def save_json(output, target_path):
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def load_json(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        return json.load(f)
