# Heimdall Frontend

Next.js frontend for Heimdall cybersecurity co-pilot.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Run development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Features

- **Minimalist UI**: Two main entrypoints (Scan Website, Audit Codebase)
- **Real-time Results**: Live scan/audit results with actionable findings
- **Report Export**: Download HTML/PDF reports
- **Plain Language**: No technical jargon, clear explanations

## Project Structure

```
app/
  layout.tsx      # Root layout
  page.tsx         # Main page with scan/audit UI
  globals.css      # Global styles
```

## Development

- Uses Next.js 14 with App Router
- Tailwind CSS for styling
- TypeScript for type safety
- Axios for API calls

## Build

```bash
npm run build
npm start
```

