import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  define: {
    'process.env': {
      NODE_ENV: '"production"'
    }
  },
  build: {
    lib: {
      entry: path.resolve(__dirname, 'src/components/index.js'),
      name: 'ReactComponents',
      formats: ['umd'],
      fileName: () => `react-components.umd.js`,
    },
    rollupOptions: {
      external: ['react', 'react-dom', 'recharts'],
      output: {
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM',
          'recharts': 'Recharts'
        },
        inlineDynamicImports: true
      },
    },
  },
});