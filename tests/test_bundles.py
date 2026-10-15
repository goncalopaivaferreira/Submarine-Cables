from __future__ import annotations

import copy
from pathlib import Path
import unittest

from arc_student import BundleValidationError, load_bundle, validate_bundle


ROOT = Path(__file__).resolve().parents[1]


class BundleTests(unittest.TestCase):
    def test_supplied_bundles_are_valid_and_comparable(self):
        bundles = [load_bundle(path) for path in sorted((ROOT / "sample_bundles").glob("*.json"))]
        self.assertEqual(len(bundles), 2)
        scenarios = {
            (bundle["episode"]["scenario"]["id"], bundle["episode"]["scenario"]["world_seed"])
            for bundle in bundles
        }
        policies = {bundle["episode"]["policy"]["id"] for bundle in bundles}
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(len(policies), 2)

    def test_wrong_schema_is_rejected(self):
        bundle = load_bundle(next((ROOT / "sample_bundles").glob("*.json")))
        changed = copy.deepcopy(bundle)
        changed["schema_version"] = "future-version"
        with self.assertRaisesRegex(BundleValidationError, "Unsupported schema"):
            validate_bundle(changed)

    def test_sensitive_payload_keys_are_rejected(self):
        bundle = load_bundle(next((ROOT / "sample_bundles").glob("*.json")))
        changed = copy.deepcopy(bundle)
        changed["episode"]["observations"] = [1, 2, 3]
        with self.assertRaisesRegex(BundleValidationError, "Forbidden key"):
            validate_bundle(changed)


if __name__ == "__main__":
    unittest.main()

