import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/smmg/AAM')
sys.path.insert(0, str(ROOT))
from AI_models.qwen import load_json, qwen3_inference

IMAGE_PATH = ROOT / 'test3.png'
PROMPT_PATH = ROOT / 'config/prompt/screening.json'
OUTPUT_PATH = ROOT / 'test_qwen_defect_result.json'
MODEL_NAME = os.getenv('QWEN_SCREENING_MODEL', 'qwen3.7-plus')
MAX_TOKENS = int(os.getenv('QWEN_SCREENING_MAX_TOKENS', '512'))


def main():
    message = load_json(PROMPT_PATH)
    message[1]['content'][0]['image'] = str(IMAGE_PATH)
    output = qwen3_inference(message, model_name=MODEL_NAME, max_tokens=MAX_TOKENS)

    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n===== Qwen raw response =====')
    for item in output:
        print(item)

    if output:
        text = str(output[0])
        start, end = text.find('['), text.rfind(']')
        if start >= 0 and end > start:
            try:
                defects = json.loads(text[start:end + 1])
                print(f'\n===== Detected defects: {len(defects)} =====')
                for i, defect in enumerate(defects, 1):
                    print(f"{i}. object_name={defect.get('object_name', '')}, confidence={defect.get('confidence', '')}")
                    print(f"   description={defect.get('description', '')}")
            except json.JSONDecodeError:
                print('\nResponse is not a valid JSON array; inspect raw output above.')

    print(f'\nSaved response to: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
