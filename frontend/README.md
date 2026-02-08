# NG12 Cancer Risk Assessor - Frontend

Simple React frontend for clinical risk assessment.

## Features

- **Assessment Form**: Enter patient details for risk assessment
- **Chat Interface**: Conversational assessment through guided questions  
- **Search**: Search NG12 guidelines directly

## Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend runs on http://localhost:3000

## Requirements

- Node.js 18+
- Backend API running on http://localhost:8000

## Configuration

The frontend proxies API calls to the backend. Configure in `vite.config.js`:

```js
proxy: {
  '/api': {
    target: 'http://localhost:8000',  // Change if needed
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, '')
  }
}
```

## Build for Production

```bash
npm run build
```

Output in `dist/` folder.

## Project Structure

```
frontend/
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── main.jsx      # Entry point
    ├── App.jsx       # Main app with all components
    └── index.css     # Styles
```
