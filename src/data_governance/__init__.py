"""EVA Data Governance foundation.

This package provides persistent identity and audit primitives. It does not
move, upload, expose, or delete user files.
"""

from .identity_store import (
    CLASSIFICATIONS,
    authorize_data_action,
    sha256_file,
    register_import,
    recover_location,
    get_record,
    list_records,
    update_location,
    record_revision,
    approve_content_revision,
    set_classification,
    set_permissions,
    set_training_approval,
    list_versions,
    list_audit_events,
    list_reconciliation_events,
    record_governance_event,
    record_reconciliation_event,
)

__all__ = [
    "CLASSIFICATIONS",
    "authorize_data_action",
    "sha256_file",
    "register_import",
    "recover_location",
    "get_record",
    "list_records",
    "update_location",
    "record_revision",
    "approve_content_revision",
    "set_classification",
    "set_permissions",
    "set_training_approval",
    "list_versions",
    "list_audit_events",
    "list_reconciliation_events",
    "record_governance_event",
    "record_reconciliation_event",
]
