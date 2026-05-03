# TB-Guard Frontend

React + Vite frontend for the TB-Guard FastAPI backend.

## 1) Install

```bash
cd frontend
npm install
```

## 2) Configure backend URL

Create `.env` in `frontend/`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

For deployed backend, replace with your public API URL.

## 3) Run

```bash
npm run dev
```

Open the shown URL (usually `http://localhost:5173`).

## 4) Build for deployment

```bash
npm run build
```

Deploy the generated `frontend/dist` folder to Vercel/Netlify/Cloudflare Pages/static host.
