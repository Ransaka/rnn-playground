from huggingface_hub import HfApi

def post_process(decode, preds):
    encodings = []
    for pred in preds:
        #only considering >0 tokens
        if pred==0:
            pass
        elif not encodings:
            encodings.append(pred)
        elif encodings[-1] != pred:
            encodings.append(pred)
    return decode(encodings)

def upload_to_hub(file_name, token, commit_message):
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name,
        repo_type='dataset',
        repo_id="Ransaka/CRNN-Sinhala-Artifacts",
        commit_message=commit_message
    )