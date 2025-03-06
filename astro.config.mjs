// @ts-check
import { defineConfig } from "astro/config";

import tailwindcss from "@tailwindcss/vite";

import react from "@astrojs/react";

// https://astro.build/config
export default defineConfig({
  vite: {
    plugins: [tailwindcss()],
  },
  server: {
    host: "0.0.0.0", // Esto permite que el servidor sea accesible desde cualquier IP en tu red local
    port: 4321, // O el puerto que desees usar
  },

  integrations: [react()],
});
