# Contributing to Heimdall

Thank you for your interest in contributing to Heimdall! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/heimdall.git`
3. Follow the setup instructions in the main README
4. Create a branch for your feature: `git checkout -b feature/your-feature-name`

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

## Integrating Agents

### Hound Integration
1. Study Hound's API/CLI interface
2. Update `run_hound_scan()` in `backend/main.py`
3. Ensure output matches expected format (see `docs/INTEGRATION.md`)
4. Test with real repositories

### BRAMA Integration
1. Study BRAMA's API/CLI interface
2. Update `run_brama_scan()` in `backend/main.py`
3. Ensure output matches expected format
4. Test with real URLs

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

