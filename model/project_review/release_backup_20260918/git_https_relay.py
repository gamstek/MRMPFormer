"""Temporary loopback relay for Git's Windows HTTPS transport failure.

Only the authorized repository is forwarded, using Python's verified HTTPS.
Credentials are held in memory, never logged or placed on the command line.
"""
import http.server
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import base64

environment = os.environ.copy()
environment.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
credential = subprocess.run(
    ['git', '-c', 'credential.interactive=never', 'credential', 'fill'],
    input='protocol=https\nhost=github.com\n\n', text=True,
    capture_output=True, env=environment, timeout=20)
fields = dict(line.split('=', 1) for line in credential.stdout.splitlines() if '=' in line)
authorization = None
if fields.get('username') and fields.get('password'):
    raw = (fields['username'] + ':' + fields['password']).encode()
    authorization = 'Basic ' + base64.b64encode(raw).decode()
print('GitHub cached credential available:', authorization is not None, flush=True)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def forward(self):
        if not self.path.startswith('/gamstek/MRMPFormer.git/'):
            self.send_error(403)
            return
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        headers = {key: value for key, value in self.headers.items()
                   if key.lower() in {'accept', 'content-type', 'git-protocol', 'user-agent'}}
        if authorization:
            headers['Authorization'] = authorization
        request = urllib.request.Request('https://github.com' + self.path,
                                         data=body if self.command == 'POST' else None,
                                         headers=headers, method=self.command)
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as error:
            response = error
        except Exception as error:
            self.send_error(502, type(error).__name__)
            return
        with response:
            payload = response.read()
            self.send_response(response.status)
            self.send_header('Content-Type', response.headers.get('Content-Type', 'application/octet-stream'))
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    do_GET = forward
    do_POST = forward


server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    command = ['git', '-c', f'url.http://127.0.0.1:{server.server_port}/.insteadOf=https://github.com/',
               '-c', 'http.postBuffer=52428800', '-c', 'credential.interactive=never'] + sys.argv[1:]
    result = subprocess.run(command, env=environment)
finally:
    server.shutdown()
sys.exit(result.returncode)
