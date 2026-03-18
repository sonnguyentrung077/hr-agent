import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backend = env.VITE_BACKEND_URL || "http://localhost:8001";

  return {
    plugins: [react()],
    server: {
      port: 3000,
      proxy: {
        "/offer": {
          target: backend,
          changeOrigin: true,
          secure: false,
          headers: {
            "ngrok-skip-browser-warning": "true",
          },
        },
        "/ws": {
          target: backend.replace(/^http/, "ws"),
          ws: true,
          changeOrigin: true,
          secure: false,
          headers: {
            "ngrok-skip-browser-warning": "true",
          },
        },
      },
    },
  };
});
