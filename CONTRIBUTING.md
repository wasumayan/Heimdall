# Contributing to Heimdall

Thank you for your interest in contributing to Heimdall! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository: https://github.com/wasumayan/Heimdall
2. Clone your fork: `git clone https://github.com/your-username/Heimdall.git`
3. Follow the setup instructions in the main README
4. Set up your `.env` file with `XAI_API_KEY` (see CONFIGURATION.md)
5. Create a branch for your feature: `git checkout -b feature/your-feature-name`

## Code Style

### Python (Backend)
- Follow PEP 8 style guide
- Use type hints where possible
- Document functions with docstrings
- Maximum line length: 100 characters

### TypeScript/React (Frontend)
- Use TypeScript for type safety
- Follow React best practices
- Use functional components with hooks
- Keep components small and focused

## Adding New Features

### Backend
1. Add new endpoints in `backend/main.py`
2. Update API documentation in `backend/README.md`
3. Add tests if applicable

### Frontend
1. Create new components in `frontend/components/` if reusable
2. Update pages in `frontend/app/`
3. Maintain the minimalist UI philosophy

## Current Integration Status

### Hound Integration ✅
- ✅ Fully integrated via subprocess calls
- ✅ Uses CLI commands: `project create`, `graph build`, `agent audit`, `finalize`
- ✅ **Whitelist Builder**: Auto-generates file whitelists within LOC budget
- ✅ Reads findings from `~/.hound/projects/{name}/hypotheses.json`
- ✅ Output transformed to Heimdall format (all fields preserved)
- ✅ **Telemetry Integration**: Real-time event streaming support
- ✅ **Graph Visualization**: Knowledge graph data API endpoints
- ✅ Virtual environment isolation

### BRAMA Integration ✅
- ✅ Fully integrated via subprocess calls
- ✅ Uses wrapper script: `agents/brama/scan_url.py`
- ✅ Includes comprehensive red-team scanner
- ✅ Output transformed to Heimdall format
- ✅ Virtual environment isolation

**Note**: Both agents are already integrated. If you want to enhance them:
1. Study the agent's CLI/API interface
2. Update wrapper functions in `backend/main.py`
3. Ensure output matches expected format
4. Test thoroughly

## Testing

Before submitting a PR:
- Test all new features
- Ensure existing functionality still works
- Check for linting errors
- Test both frontend and backend

## Pull Request Process

1. Update documentation if needed
2. Add comments for complex logic
3. Write a clear PR description
4. Reference any related issues
5. Ensure CI checks pass (if configured)

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about implementation
- Documentation improvements

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Maintain a positive environment

