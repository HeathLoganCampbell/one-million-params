// Minimal static server. `python -m http.server` works too, but this sets the
// correct MIME type for .wasm (needed for streaming instantiation).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const port = Number(process.env.PORT) || 8080;

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".bin": "application/octet-stream",
};

http
  .createServer((req, res) => {
    const url = decodeURIComponent((req.url || "/").split("?")[0]);
    const rel = url === "/" ? "index.html" : url.replace(/^\/+/, "");
    const file = path.resolve(root, rel);

    if (!file.startsWith(root)) {
      res.writeHead(403).end("forbidden");
      return;
    }

    fs.readFile(file, (err, data) => {
      if (err) {
        res.writeHead(404).end("not found");
        return;
      }
      res.writeHead(200, {
        "content-type": TYPES[path.extname(file)] || "application/octet-stream",
        "cache-control": "no-cache",
      });
      res.end(data);
    });
  })
  .listen(port, () => {
    console.log(`tiny-gpt → http://localhost:${port}`);
  });
