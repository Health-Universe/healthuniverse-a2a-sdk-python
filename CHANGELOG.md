# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-01

### Added
- Initial release of Health Universe A2A SDK
- `A2AAgent` base class with simple API
- `StreamingAgent` for short-running streaming tasks
- `AsyncAgent` for long-running background tasks
- `MessageContext` and `AsyncContext` for agent processing
- `ValidationResult` for message validation
- Automatic extension configuration (file access, background jobs)
- Progress update methods (`context.update_progress()`)
- Artifact generation methods (`context.add_artifact()`)
- Lifecycle hooks (on_startup, on_shutdown, on_task_start, etc.)
- Comprehensive documentation and examples
- Unit tests for core functionality
- Example agents (calculator, data analyzer, file processor, batch processor)
- Type hints throughout the codebase
- Development tooling (ruff, mypy, pytest)

### Features
- SDK Method Calls API (not generator pattern)
- Validation before job enqueueing (AsyncAgent)
- File access extension support
- Background job extension support
- Cancellation checking
- Error handling with custom messages
- Configurable timeouts (AsyncAgent)
- MIME type declarations
- Agent versioning

[Unreleased]: https://github.com/Health-Universe/healthuniverse-a2a-sdk-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Health-Universe/healthuniverse-a2a-sdk-python/releases/tag/v0.1.0
