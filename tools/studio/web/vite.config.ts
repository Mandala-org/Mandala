import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // STUDIO_API points the development server at another backend port.
  server: { port: 5173, proxy: { "/api": process.env.STUDIO_API ?? "http://127.0.0.1:8765" } },
  build: { outDir: "dist", chunkSizeWarningLimit: 2000 },
});
