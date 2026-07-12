/**
 * BioLayer dashboard — zero-dependency static server.
 *
 * Serves ./public on http://localhost:PORT. No npm install required; this is
 * intentionally plain `http` + `fs` so the demo laptop never depends on a
 * registry being reachable during the hackathon.
 *
 * Usage:
 *   node server.js            # http://localhost:4173
 *   PORT=8080 node server.js  # custom port
 */
const http = require("http");
const fs = require("fs");
const path = require("path");
const { execFile } = require("child_process");

const PORT = Number(process.env.PORT) || 4173;
const PUBLIC_DIR = path.join(__dirname, "public");
const REPO_ROOT = path.join(__dirname, "..");
const PYTHON = process.env.PYTHON || "python3";
const BRIDGE = path.join(__dirname, "bridge.py");

// Run the Python bridge (real certify infra) and return parsed JSON. Any failure ->
// reject, and the caller answers 503 so the front-end keeps its static mock globals.
function runBridge(args, cb) {
  execFile(PYTHON, [BRIDGE, ...args],
    { cwd: REPO_ROOT, timeout: 240000, maxBuffer: 32 * 1024 * 1024,
      env: { ...process.env, PYTHONPATH: REPO_ROOT } },
    (err, stdout) => {
      if (err && !stdout) return cb(err);
      let json;
      try { json = JSON.parse(stdout); } catch (e) { return cb(e); }
      if (json && json.error) return cb(new Error(json.error));
      cb(null, json);
    });
}

function sendJson(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json; charset=utf-8",
                        "Cache-Control": "no-store",
                        // allow the UI to be hosted on a different origin (e.g. served
                        // from the laptop while the API is a forwarded SageMaker port)
                        "Access-Control-Allow-Origin": "*" });
  res.end(body);
}

// /api/* -> real certify infra via the bridge. Returns true if it handled the request.
function handleApi(req, res, urlPath) {
  if (req.method === "OPTIONS") {                 // CORS preflight for cross-origin UI
    res.writeHead(204, { "Access-Control-Allow-Origin": "*",
                         "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                         "Access-Control-Allow-Headers": "Content-Type" });
    res.end();
    return true;
  }
  if (urlPath === "/api/all" && req.method === "GET") {
    runBridge(["all"], (err, json) =>
      err ? sendJson(res, 503, { error: String(err.message || err) }) : sendJson(res, 200, json));
    return true;
  }
  if (urlPath === "/api/certify_answer" && req.method === "POST") {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      let p = {};
      try { p = body ? JSON.parse(body) : {}; } catch (e) { /* use defaults */ }
      const args = ["certify_answer"];
      if (p.prompt) args.push("--prompt", String(p.prompt));
      if (p.answer) args.push("--answer", String(p.answer));
      if (p.track) args.push("--track", String(p.track));
      if (p.bedrock) args.push("--bedrock");
      runBridge(args, (err, json) =>
        err ? sendJson(res, 503, { error: String(err.message || err) }) : sendJson(res, 200, json));
    });
    return true;
  }
  return false;
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

const server = http.createServer((req, res) => {
  let urlPath = decodeURIComponent((req.url || "/").split("?")[0]);

  // real certify infra endpoints take precedence over static files
  if (urlPath.startsWith("/api/") && handleApi(req, res, urlPath)) return;

  if (urlPath === "/") urlPath = "/index.html";

  const filePath = path.normalize(path.join(PUBLIC_DIR, urlPath));
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403, { "Content-Type": "text/plain" });
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end(`Not found: ${urlPath}`);
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`BioLayer dashboard running at http://localhost:${PORT} (bound 0.0.0.0)`);
  console.log(`Serving ${PUBLIC_DIR}  ·  API: /api/all, /api/certify_answer`);
  console.log(`Forward this port to your laptop, then open http://localhost:${PORT}`);
});
