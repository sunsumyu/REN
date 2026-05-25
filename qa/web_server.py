import http.server
import json
import logging
import urllib.parse
import os
import sys

# Ensure we can import prompts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import prompts

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MixedHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # We specify the directory to serve static files from
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Add basic headers for convenience
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        # Handle CORS preflight requests
        self.send_response(200, "OK")
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        # 1. API: List all prompts with active status
        if path == "/api/prompts":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            
            try:
                prompts_list = []
                for name in prompts.PROMPT_NAMES:
                    history = prompts.list_prompt_versions(name)
                    # Find active version
                    active_version = None
                    active_desc = ""
                    for v in history:
                        if v["is_active"]:
                            active_version = v["version"]
                            active_desc = v["description"]
                            break
                    
                    prompts_list.append({
                        "name": name,
                        "active_version": active_version,
                        "description": active_desc,
                        "total_versions": len(history)
                    })
                
                response = {"success": True, "data": prompts_list}
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to fetch prompts: {e}")
                self.send_error_response(500, f"Failed to fetch prompts: {e}")
                
        # 2. API: Fetch version history for a specific prompt
        elif path == "/api/prompts/history":
            query = urllib.parse.parse_qs(parsed_url.query)
            prompt_name = query.get("name", [None])[0]
            
            if not prompt_name or prompt_name not in prompts.PROMPT_NAMES:
                self.send_error_response(400, "Missing or invalid 'name' parameter.")
                return
                
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            
            try:
                history = prompts.list_prompt_versions(prompt_name)
                response = {"success": True, "data": history}
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to fetch history for {prompt_name}: {e}")
                self.send_error_response(500, f"Error: {e}")
                
        else:
            # Fallback to standard static file serving
            super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        # 1. API: Update a prompt and save a new version
        if path == "/api/prompts/update":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")
            
            try:
                data = json.loads(post_data)
                name = data.get("name")
                content = data.get("content")
                description = data.get("description", "Updated via Web Console")
                
                if not name or not content:
                    self.send_error_response(400, "Missing required fields: 'name' or 'content'.")
                    return
                    
                if name not in prompts.PROMPT_NAMES:
                    self.send_error_response(400, f"Invalid prompt name: {name}")
                    return
                
                # Perform the update
                new_version = prompts.update_prompt(name, content, description)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                
                response = {
                    "success": True, 
                    "msg": f"Successfully created Version {new_version}.",
                    "data": {"name": name, "version": new_version}
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to update prompt: {e}")
                self.send_error_response(500, f"Failed to save prompt: {e}")
                
        # 2. API: Rollback to a specific prompt version
        elif path == "/api/prompts/rollback":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")
            
            try:
                data = json.loads(post_data)
                name = data.get("name")
                version = data.get("version")
                
                if not name or version is None:
                    self.send_error_response(400, "Missing required fields: 'name' or 'version'.")
                    return
                    
                if name not in prompts.PROMPT_NAMES:
                    self.send_error_response(400, f"Invalid prompt name: {name}")
                    return
                
                try:
                    version = int(version)
                except ValueError:
                    self.send_error_response(400, "Version must be an integer.")
                    return
                
                # Perform rollback
                success = prompts.rollback_prompt(name, version)
                
                if success:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    
                    response = {
                        "success": True,
                        "msg": f"Prompt '{name}' successfully rolled back to Version {version}."
                    }
                    self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
                else:
                    self.send_error_response(400, f"Failed to rollback. Version {version} may not exist.")
            except Exception as e:
                logger.error(f"Failed to rollback prompt: {e}")
                self.send_error_response(500, f"Internal Error: {e}")
                
        else:
            self.send_error_response(404, "Endpoint not found.")

    def send_error_response(self, code: int, message: str):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        response = {"success": False, "msg": message}
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))

def run_server():
    server_address = ("", PORT)
    httpd = http.server.HTTPServer(server_address, MixedHTTPRequestHandler)
    logger.info(f"============================================================")
    logger.info(f"   Prompt Web Console Server is running on port {PORT}")
    logger.info(f"   Open: http://localhost:{PORT}/index.html in your browser")
    logger.info(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server is stopping...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
