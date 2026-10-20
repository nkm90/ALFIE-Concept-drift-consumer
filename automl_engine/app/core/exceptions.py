class AutoMLError(Exception):
    """Base exception for all AutoML pipeline errors."""


class AutoMLValidationError(AutoMLError, ValueError):
    """Raised when caller-supplied inputs are invalid."""


class AutoMLDataError(AutoMLValidationError):
    """Raised for DataFrame-level problems."""


class AutoMLConfigError(AutoMLValidationError):
    """Raised when a configuration value or constant is out of range."""


class AutoMLRuntimeError(AutoMLError, RuntimeError):
    """Raised when a pipeline step fails at runtime."""


class AutoMLTrainingError(AutoMLRuntimeError):
    """Raised when AutoGluon's .fit() call fails."""


class AutoMLDeploymentError(AutoMLRuntimeError):
    """Raised when cloning or loading the deployment predictor fails."""


class AutoMLLeaderboardError(AutoMLRuntimeError):
    """Raised when leaderboard generation fails."""


class AutoDWDownloadError(AutoMLRuntimeError):
    """Raised when downloading a dataset or resource fails."""


class AutoDWUploadError(AutoMLRuntimeError):
    """Raised when uploading a model or artifact fails.

    ``status_code`` carries the HTTP status a router should report for the
    failure. It defaults to ``502`` (AutoDW communication failure); when the
    upload completed but AutoDW itself returned an error response, the
    orchestrator sets it to the upstream status so that status propagates to
    the caller.
    """

    def __init__(self, message: str = "", *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class AutoMLSerializationError(AutoMLRuntimeError):
    """Raised when serializing or pickling a model fails."""


class AutoMLImageError(AutoMLRuntimeError):
    """Raised when image conversion or processing fails."""


class AutoMLChatError(AutoMLRuntimeError):
    """Raised when an LLM/VLM call fails."""
