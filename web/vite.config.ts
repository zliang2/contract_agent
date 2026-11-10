// Dev server proxies API calls to uvicorn on :8000; in production FastAPI serves dist/.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const api = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/query": api, "/contracts": api, "/healthz": api } },
  test: { environment: "happy-dom", globals: true, setupFiles: ["src/setupTests.ts"] },
});
