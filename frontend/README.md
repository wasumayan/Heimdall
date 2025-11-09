# Heimdall Frontend

Next.js frontend for Heimdall cybersecurity co-pilot.

## Current Status: Complete ✅

- ✅ Full UI implementation
- ✅ Two main entrypoints (Scan Website, Audit Codebase)
- ✅ Real-time results display
- ✅ Report export functionality
- ✅ Responsive design
- ✅ Error handling
- ✅ **Whitelist Configuration Modal**: Popup interface for file whitelist setup
- ✅ **Always-Visible Whitelist Card**: Status display with quick configure button
- ✅ **All Hound CLI Options**: Complete audit parameter control
- ✅ **Interactive Graph Visualization**: D3.js knowledge graph viewer
- ✅ **Telemetry Dashboard**: Real-time audit monitoring panel
- ✅ **Complete Findings Display**: All Hound fields (evidence, reasoning, node refs, etc.)

## Setup

1. Install dependencies:
```bash
npm install
```

2. (Optional) Create `.env.local` if you need to change API URL:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Note**: Defaults to `http://localhost:8000` if not set.

3. Run development server:
```bash
npm run dev
# Or: PORT=3001 npm run dev (if port 3000 is in use)
```

The app will be available at `http://localhost:3000` (or `http://localhost:3001` if 3000 is occupied)

## Features

- **Minimalist UI**: Two main entrypoints (Scan Website, Audit Codebase)
- **Real-time Results**: Live scan/audit results with actionable findings
- **Report Export**: Download HTML/PDF reports
- **Plain Language**: No technical jargon, clear explanations
- **Whitelist Configuration**: 
  - Auto-generation with LOC budget (default: 50,000)
  - Manual file list override option
  - Always-visible status card
  - Easy-to-use modal interface
- **Advanced Audit Options**: All Hound CLI parameters exposed
- **Graph Visualization**: Interactive D3.js knowledge graph viewer
- **Telemetry**: Real-time audit event streaming (optional)
- **Complete Findings**: All Hound output fields displayed (evidence, reasoning, node refs, etc.)

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

