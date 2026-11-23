"""EVA Data Governance foundation.

This package provides persistent identity and audit primitives. It does not
move, upload, expose, or delete user files.
"""

from .identity_store import (
    CLASSIFICATIONS,
    register_import,
    get_record,
    list_records,
    update_location,
    record_revision,
    set_classification,
    set_permissions,
    set_training_approval,
    list_versions,
    list_audit_events,
)

__all__ = [
    "CLASSIFICATIONS",
    "register_import",
    "get_record",
    "list_records",
    "update_location",
    "record_revision",
    "set_classification",
    "set_permissions",
    "set_training_approval",
    "list_versions",
    "list_audit_events",
]
