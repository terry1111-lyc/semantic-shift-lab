import tempfile
import unittest
from pathlib import Path
from semantic_shift.identity import directory_digest, validate_model_identity


class IdentityTests(unittest.TestCase):
    def test_local_model_change_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'config.json').write_text('{}')
            digest=directory_digest(path)
            validate_model_identity(str(path),digest)
            (path/'config.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError): validate_model_identity(str(path),digest)

    def test_hub_revision_must_be_pinned(self):
        validate_model_identity('org/model','a'*40)
        for revision in ('main','v1','z'*40,'a'*39):
            with self.assertRaises(ValueError): validate_model_identity('org/model',revision)

    def test_empty_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): directory_digest(tmp)
