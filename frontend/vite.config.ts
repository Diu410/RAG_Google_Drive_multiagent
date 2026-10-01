import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
  plugins:[react()],
  server:{
    // Cargo writes and locks Windows executables under src-tauri/target while
    // `tauri dev` is running. They are never frontend inputs, and asking the
    // Vite watcher to open them can terminate dev startup with EBUSY.
    watch:{ignored:['**/src-tauri/target/**','**/src-tauri/binaries/**']},
    proxy:{'/api':{target:'http://127.0.0.1:8000',rewrite:path=>path.replace(/^\/api/,'')}},
  },
});
