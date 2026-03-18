import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/offer": {
        target: process.env.VITE_BACKEND_URL || "http://localhost:8001",
        changeOrigin: true,
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      },
      "/ws": {
        target: (process.env.VITE_BACKEND_URL || "http://localhost:8001").replace(
          /^http/,
          "ws"
        ),
        ws: true,
        changeOrigin: true,
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      },
    },
  },
});
