from huggingface_hub import HfApi, login

print("=== Hugging Face Model Uploader ===")
token = input("Please paste your Hugging Face Access Token here and press Enter: ")

print("\nLogging in...")
login(token=token.strip())

print("\nUploading model.pt... This might take a few minutes depending on your internet speed.")
api = HfApi()
api.upload_file(
    path_or_fileobj="mental_health_bert_bilstm_model/model.pt",
    path_in_repo="model.pt",
    repo_id="sivvvsivagami/mindguard-ai-models",
    repo_type="model"
)
print("\n✅ Upload completely successful! You can now deploy on Streamlit Cloud.")
