
from google_auth_oauthlib.flow import InstalledAppFlow #св
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os

def get_google_drive_credentials(credentials_path: str, token_path: str = 'token.json', interactive: bool = False) -> Credentials:
    """Генерирует или загружает учетные данные для Google Drive API."""
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)


    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not interactive:
                raise RuntimeError('Google Drive authorization required. Run api/gdrive_auth.py.')
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
            creds = flow.run_local_server(port=0)

        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    return creds


if __name__ == '__main__':
    from pathlib import Path
    base = Path(__file__).resolve().parent
    get_google_drive_credentials(str(base / 'credentials.json'), str(base / 'token.json'), interactive=True)
