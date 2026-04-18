# AI Agent Framework - Testing Makefile
# Quick commands to test your APIs after code changes

.PHONY: help test test-quick test-manual test-health test-full install-test-deps

# Default target
help:
	@echo "AI Agent Framework - Testing Commands"
	@echo "===================================="
	@echo ""
	@echo "Quick Testing:"
	@echo "  make test-health    - Quick health check (30 seconds)"
	@echo "  make test-quick     - Fast API validation (2-3 minutes)"
	@echo "  make test-manual    - Comprehensive manual tests (5-10 minutes)"
	@echo "  make test-full      - Full automated test suite (10-15 minutes)"
	@echo ""
	@echo "Setup:"
	@echo "  make install-test-deps - Install testing dependencies"
	@echo ""
	@echo "Configuration:"
	@echo "  Set BASE_URL to test different servers:"
	@echo "  make test-health BASE_URL=http://your-server:8000"

# Variables
BASE_URL ?= http://localhost:8000
PYTHON ?= python
PYTEST_ARGS ?= -v --tb=short

# Install testing dependencies
install-test-deps:
	@echo "Installing testing dependencies..."
	poetry install --with test
	@echo "✅ Testing dependencies installed"

# Quick health check (fastest)
test-health:
	@echo "🏥 Running health check..."
	$(PYTHON) tests/utils/health_check.py --base-url $(BASE_URL)

# Quick API tests (medium speed)
test-quick:
	@echo "🚀 Running quick API tests..."
	$(PYTHON) tests/manual/test_api_manual.py --base-url $(BASE_URL)

# Manual comprehensive tests
test-manual:
	@echo "🔧 Running comprehensive manual tests..."
	$(PYTHON) tests/manual/test_api_manual.py --base-url $(BASE_URL) --verbose

# Full automated test suite
test-full:
	@echo "🧪 Running full test suite..."
	TEST_BASE_URL=$(BASE_URL) poetry run pytest tests/integration/test_api_endpoints.py $(PYTEST_ARGS)

# Alias for most common usage
test: test-quick

# Development workflow targets
dev-check: test-health test-quick
	@echo "✅ Development checks complete"

# CI/CD workflow target
ci-test: install-test-deps test-full
	@echo "✅ CI tests complete"

# Clean up any test artifacts
clean:
	@echo "🧹 Cleaning up test artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Clean complete"
