"""Скачивает веса multilingual-e5-small один раз. Дальше все работает без интернета."""
from pathlib import Path
import urllib.request

MODEL_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
FILES = ['config.json', 'model.safetensors', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'sentencepiece.bpe.model']

if __name__ == '__main__':
    folder = Path(__file__).resolve().parent / 'models' / 'multilingual-e5-small'
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = folder / name
        if path.exists():
            continue
        print('Downloading', name, flush=True)
        url = f'https://huggingface.co/intfloat/multilingual-e5-small/resolve/{MODEL_REVISION}/{name}'
        # Качаем во временный файл, чтобы оборванная загрузка не выглядела как готовая
        temporary = path.with_suffix(path.suffix + '.download')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    (folder / 'source.txt').write_text(
        f'https://huggingface.co/intfloat/multilingual-e5-small\nRevision: {MODEL_REVISION}\nLicense: MIT\n',
        encoding='utf-8')
    print('Model ready:', folder)
