import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// PROJECT_PLAN.md section 12: the clinician dashboard talks to clinician-api
// (the BFF, port 8007). Proxied here so the dev server and the API share an
// origin -- no CORS configuration needed on clinician-api for local dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8007",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/ws": {
        target: "ws://localhost:8006",
        ws: true,
      },
    },
  },
});
