"""Session context utilities for Pinocchio."""
import threading

_current_session = threading.local()


def set_current_session(session):
    """Set the current session object."""
    _current_session.value = session


def get_current_session():
    """Get the current session object."""
    return getattr(_current_session, "value", None)
