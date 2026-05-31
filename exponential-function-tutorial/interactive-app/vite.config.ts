import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: './' 使构建产物用相对路径，可直接 file:// 打开或托管到任意子路径 / GitHub Pages
export default defineConfig({
  plugins: [react()],
  base: './',
})
