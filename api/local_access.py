"""Local API access and request limits, before routing or form parsing."""
import hmac
import json
import os
import secrets
from urllib.parse import urlsplit

from starlette.exceptions import HTTPException

from settings import MAX_UPLOAD_BYTES

API_TOKEN = os.environ.get('KONTUR_API_TOKEN') or secrets.token_urlsafe(32)
ALLOWED_ORIGINS = {
    'http://tauri.localhost', 'https://tauri.localhost', 'tauri://localhost',
    'http://127.0.0.1:5173', 'http://localhost:5173',
}


class LocalAccessMiddleware:
    def __init__(self, app, *, token=None, desktop=None, allowed_hosts=None):
        self.app = app
        self.token = token or API_TOKEN
        self.desktop = os.getenv('KONTUR_DESKTOP') == '1' if desktop is None else desktop
        self.allowed_hosts = set(allowed_hosts or ('127.0.0.1', 'localhost', '::1'))
        self.json_limit = int(os.getenv('MAX_JSON_REQUEST_BYTES', str(8 * 1024 * 1024)))

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = {key.lower(): value.decode('latin1') for key, value in scope['headers']}

        async def reject(status, detail):
            body = json.dumps({'detail': detail}).encode()
            await send({'type': 'http.response.start', 'status': status,
                        'headers': [(b'content-type', b'application/json'), (b'cache-control', b'no-store')]})
            await send({'type': 'http.response.body', 'body': body})

        host = headers.get(b'host', '')
        try:
            hostname = urlsplit('http://' + host).hostname
        except ValueError:
            hostname = None
        if hostname not in self.allowed_hosts:
            return await reject(400, 'Invalid local API host')
        origin = headers.get(b'origin')
        if origin and origin not in ALLOWED_ORIGINS and origin != f"{scope.get('scheme', 'http')}://{host}":
            return await reject(403, 'Origin is not allowed')
        if scope['method'] == 'OPTIONS':
            return await self.app(scope, receive, send)
        path = scope['path']
        if path == '/access-token':
            if self.desktop or scope['method'] != 'GET' or headers.get(b'x-fortuna-client') != 'web':
                return await reject(403, 'API access must be initialized by the application')
            body = json.dumps({'token': self.token}).encode()
            await send({'type': 'http.response.start', 'status': 200,
                        'headers': [(b'content-type', b'application/json'), (b'cache-control', b'no-store'),
                                    (b'x-content-type-options', b'nosniff')]})
            return await send({'type': 'http.response.body', 'body': body})
        public = scope['method'] in {'GET', 'HEAD'} and (
            path in {'/', '/api/health', '/favicon.ico'} or path.startswith('/assets/'))
        supplied = headers.get(b'x-fortuna-token', '')
        if not public and not hmac.compare_digest(supplied.encode(), self.token.encode()):
            return await reject(401, 'Initialize API access or restart the application')
        upload = path in {'/upload-doc', '/transcribe'}
        if upload and scope['method'] == 'POST' and not headers.get(b'content-type', '').lower().startswith('multipart/form-data;'):
            return await reject(415, 'Use multipart/form-data for uploads')
        limit = MAX_UPLOAD_BYTES + 1024 * 1024 if upload else self.json_limit
        try:
            length = int(headers.get(b'content-length', '0'))
        except ValueError:
            return await reject(400, 'Invalid content length')
        if length < 0 or length > limit:
            return await reject(413, 'Request body exceeds size limit')
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > limit:
                    raise HTTPException(413, 'Request body exceeds size limit')
            return message

        await self.app(scope, limited_receive, send)
