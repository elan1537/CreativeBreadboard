import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// https://vitejs.dev/config/
export default defineConfig(({ command, mode, ssrBuild }) => {
  if (mode === "development") {
    return {
      server: {
        host: "0.0.0.0",
        port: 8080,
      },
      plugins: [vue()],
    };
  } else {
    return {
      plugins: [vue()],
      build: {
        minify: "terser",
        terserOptions: {
          compress: {
            drop_console: true,
            drop_debugger: true,
          },
        },
      },
    };
  }
});
