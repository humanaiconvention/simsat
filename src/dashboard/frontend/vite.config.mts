import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: "assets/main.js",
        assetFileNames: "assets/[name].[ext]",
      },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      // In dev mode: proxy /api/* to Django dashboard
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      // In dev mode: proxy /sim/* directly to simulator (bypasses Django proxy)
      "/sim": {
        target: "http://localhost:9005",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/sim/, ""),
      },
    },
  },
  base: "/static/",
});
