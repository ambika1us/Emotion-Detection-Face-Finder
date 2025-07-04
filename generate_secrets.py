import json

def json_to_secrets_toml(json_path, toml_path=".streamlit/secrets.toml"):
    with open(json_path, "r") as f:
        creds = json.load(f)

    # Ensure multiline private key formatting
    private_key = creds["private_key"].replace("\n", "\\n")

    with open(toml_path, "w", encoding="utf-8") as f:
        f.write("[google_sheets]\n")
        for key in [
            "type", "project_id", "private_key_id", "client_email",
            "client_id", "auth_uri", "token_uri", "auth_provider_x509_cert_url",
            "client_x509_cert_url"
        ]:
            f.write(f'{key} = "{creds[key]}"\n')
        f.write(f'private_key = """{creds["private_key"]}"""\n')

    print(f"✅ Secrets TOML generated at: {toml_path}")

# Run this if needed
json_to_secrets_toml("secret/service_account.json")