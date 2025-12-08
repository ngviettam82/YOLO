#!/usr/bin/env python3
"""
Image Server for Label Studio
Serves images over HTTP so Label Studio can access them
"""

import http.server
import socketserver
import os
import sys
import mimetypes
from pathlib import Path
from urllib.parse import urlparse, unquote
import io

PORT = 8000

class ImageRequestHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP request handler for serving images with optimizations"""
    
    def end_headers(self):
        """Add CORS and cache headers to allow cross-origin requests"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-type')
        # Enable browser caching for faster navigation
        self.send_header('Cache-Control', 'public, max-age=86400')
        self.send_header('Connection', 'keep-alive')
        super().end_headers()
    
    def do_GET(self):
        """Handle GET requests efficiently"""
        # Parse the URL
        parsed_path = urlparse(self.path)
        file_path = unquote(parsed_path.path)
        
        # Remove leading slash
        if file_path.startswith('/'):
            file_path = file_path[1:]
        
        # Security: prevent directory traversal
        if '..' in file_path:
            self.send_error(403, "Access denied")
            return
        
        # Check if file exists
        full_path = os.path.join(os.getcwd(), file_path)
        
        try:
            if not os.path.exists(full_path):
                print(f"❌ File not found: {file_path}")
                self.send_error(404, "File not found")
                return
            
            if not os.path.isfile(full_path):
                self.send_error(400, "Bad request")
                return
            
            # Get file size
            file_size = os.path.getsize(full_path)
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(full_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            
            # Send response headers
            self.send_response(200)
            self.send_header('Content-type', mime_type)
            self.send_header('Content-Length', file_size)
            self.end_headers()
            
            # Send file content
            with open(full_path, 'rb') as f:
                self.wfile.write(f.read())
            
            print(f"✓ Served: {file_path} ({file_size} bytes)")
            
        except Exception as e:
            print(f"❌ Error serving {file_path}: {e}")
            self.send_error(500, "Internal server error")
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-type')
        self.send_header('Content-Length', '0')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def run_server():
    """Start the image server with threading for better performance"""
    print("=" * 80)
    print("🖼️  Image Server for Label Studio")
    print("=" * 80)
    print()
    print(f"📁 Serving images from: {os.getcwd()}")
    print(f"🌐 URL: http://localhost:{PORT}")
    print()
    print("✓ Server started!")
    print("✓ Label Studio can now load images from this server")
    print()
    print("⚠️  KEEP THIS WINDOW OPEN!")
    print("   Close this window to stop the server")
    print()
    print("=" * 80)
    print()
    
    try:
        # Use ThreadingTCPServer for better concurrent request handling
        class ThreadingTCPServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
        
        with ThreadingTCPServer(("", PORT), ImageRequestHandler) as httpd:
            print(f"Listening on port {PORT}...")
            print()
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48 or e.errno == 98:  # Port already in use
            print()
            print("❌ ERROR: Port already in use!")
            print()
            print("Solutions:")
            print("1. Stop the other process using port 8000")
            print("2. Edit this file and change PORT = 8000 to PORT = 8001")
            print()
            sys.exit(1)
        else:
            raise
    except KeyboardInterrupt:
        print()
        print()
        print("🛑 Server stopped")
        sys.exit(0)


if __name__ == "__main__":
    # Find the most recent label_studio_* project folder
    script_dir = Path(__file__).parent  # AutoLabel/scripts
    project_root = script_dir.parent.parent  # Go up to YOLO root
    
    # Find the most recent label_studio_* directory
    project_dirs = list(project_root.glob("label_studio_*"))
    if not project_dirs:
        print("❌ ERROR: No label_studio_* project directory found!")
        print(f"Expected to find in: {project_root}")
        sys.exit(1)
    
    # Get the most recent one
    project_dir = max(project_dirs, key=lambda p: p.stat().st_mtime)
    os.chdir(project_dir)
    
    print(f"📁 Working directory: {os.getcwd()}")
    print()
    
    run_server()
