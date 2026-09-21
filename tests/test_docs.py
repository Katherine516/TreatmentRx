"""The docs are a deliverable here, so a few of their claims are testable.

This repo's stated value is that a reader can tell a measured number from a
placeholder. That only holds while the prose agrees with the code, and it stopped
holding: `CLAUDE.md` was updated after the dWOLS cross-arm covariance was kept and
`README.md` was not, so the two documents disagreed on every headline table —
coverage 95% against 98%, abstention 67% against 78%, transfer 57-89% against
70-95% — and ten source docstrings still carried the pre-fix figures.

The abstention rate has since moved again (invariant 55), which is why the check
below compares the served card against the prose rather than against a literal:
a figure pinned in four places is four places to forget.

Most of those numbers are Monte Carlo outputs and cannot be asserted in a fast
test. These are the ones that can: the command list, the test count, and the
constants the prose quotes by value.
"""

from __future__ import annotations

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLAUDE_MD = ROOT / "CLAUDE.md"
README = ROOT / "README.md"


def _documented_commands(text: str) -> set[str]:
    return set(re.findall(r"treatmentrx\.cli ([a-z]+)", text))


class CommandListTests(unittest.TestCase):
    """A documented command that does not exist is a broken instruction."""

    def setUp(self):
        from treatmentrx import cli

        parser_commands = set()
        for action in cli.build_parser()._actions if hasattr(cli, "build_parser") else []:
            choices = getattr(action, "choices", None)
            if choices:
                parser_commands |= set(choices)
        self.parser_commands = parser_commands

    def test_every_documented_command_exists(self):
        if not self.parser_commands:
            self.skipTest("cli does not expose a reusable parser")
        for source in (CLAUDE_MD, README):
            with self.subTest(doc=source.name):
                documented = _documented_commands(source.read_text())
                unknown = documented - self.parser_commands
                self.assertFalse(unknown, f"{source.name} documents {sorted(unknown)}")

    def test_claude_md_lists_every_command(self):
        if not self.parser_commands:
            self.skipTest("cli does not expose a reusable parser")
        documented = _documented_commands(CLAUDE_MD.read_text())
        missing = self.parser_commands - documented
        self.assertFalse(missing, f"CLAUDE.md does not list {sorted(missing)}")


class QuotedConstantTests(unittest.TestCase):
    """Where the prose quotes a constant by value, it has to be that value."""

    def test_the_documented_multiplicity_family_is_the_one_applied(self):
        """15 appears throughout both docs as the number of unordered pairs."""
        from treatmentrx.arms import TREATMENT_ARMS
        from treatmentrx.estimation.inference import DEFAULT_ALPHA, simultaneous_alpha

        pairs = len(TREATMENT_ARMS) * (len(TREATMENT_ARMS) - 1) // 2
        self.assertEqual(pairs, 15)
        self.assertAlmostEqual(
            simultaneous_alpha(len(TREATMENT_ARMS)), DEFAULT_ALPHA / 15
        )
        for source in (CLAUDE_MD, README):
            with self.subTest(doc=source.name):
                self.assertIn("15 unordered pairs", source.read_text())

    def test_the_documented_deployed_training_size_is_the_real_one(self):
        from treatmentrx.estimation import training
        from treatmentrx.feedback.coverage import DEPLOYED_N

        deployed = int(training.COHORT_SIZE * (1.0 - training.HOLDOUT_FRACTION))
        self.assertEqual(deployed, DEPLOYED_N)
        for source in (CLAUDE_MD, README):
            with self.subTest(doc=source.name):
                self.assertIn(f"n={deployed}", source.read_text())

    def test_the_served_stage_indices_are_documented_as_measured(self):
        from treatmentrx.feedback import coverage

        self.assertEqual(coverage.SERVED_STAGE_INDICES, (1, 2))
        self.assertIn("SERVED_STAGE_INDICES", CLAUDE_MD.read_text())


class ZeroDependencyTests(unittest.TestCase):
    """`dependencies = []` is a hard constraint, and nothing asserted it.

    CLAUDE.md calls it deliberate — it is what keeps the prototype auditable and
    installable anywhere — and `pyproject.toml` declares it, but declaring is not
    checking. This walks every module and resolves each top-level import, which
    also pins the claim `AgentLayer`'s docstring makes: there is no model call
    here, and a package that imports only the standard library cannot acquire one
    quietly.
    """

    @classmethod
    def setUpClass(cls):
        import ast

        cls.imports = {}
        for path in sorted((ROOT / "src" / "treatmentrx").rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        cls.imports.setdefault(alias.name.split(".")[0], set()).add(path.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        cls.imports.setdefault(node.module.split(".")[0], set()).add(path.name)

    @staticmethod
    def _is_stdlib(name: str) -> bool:
        """Python 3.9 has no `sys.stdlib_module_names`, so resolve the spec and
        look at where it lives."""
        import importlib.util
        import sys
        import sysconfig

        if name in sys.builtin_module_names:
            return True
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            return False
        if spec is None or spec.origin in (None, "built-in", "frozen"):
            return spec is not None
        stdlib = sysconfig.get_paths()["stdlib"]
        return spec.origin.startswith(stdlib) and "site-packages" not in spec.origin

    def test_the_package_imports_only_the_standard_library(self):
        self.assertTrue(self.imports, "no imports were parsed, so this proves nothing")
        third_party = {
            name: sorted(files)
            for name, files in self.imports.items()
            if name != "treatmentrx" and not self._is_stdlib(name)
        }
        self.assertEqual(
            third_party,
            {},
            f"a third-party import has appeared: {third_party}",
        )

    def test_pyproject_still_declares_none(self):
        self.assertIn("dependencies = []", (ROOT / "pyproject.toml").read_text())

    def test_the_sweep_reaches_the_modules_it_claims_to(self):
        """Otherwise a path change makes this pass by parsing nothing."""
        seen = {name for files in self.imports.values() for name in files}
        for expected in ("linalg.py", "dwols.py", "rationale.py", "contract.py"):
            with self.subTest(module=expected):
                self.assertIn(expected, seen)


class InvariantIndexTests(unittest.TestCase):
    """The index at the top of the invariants has to be derived, not curated.

    68 invariants in discovery order is the right record and the wrong lookup:
    `q_values` alone appears in twelve of them. The table exists so a reader
    about to touch something can find what has already gone wrong on it — and an
    index that goes stale is worse than none, because it reads as complete.

    So the table is *checked against the invariant bodies* here. Add an
    invariant that touches one of these and this fails until the row is updated.
    """

    # The declaration is the token set; the numbers in CLAUDE.md are derived
    # from it. A token matches when it appears inside a backticked span, so
    # `linalg` catches `linalg.inverse`.
    ROWS = (
        ("`q_values`, and anything that ranks from them",
         ("q_values", "Q_FLOOR", "Q_CEILING")),
        ("`recommended_arm` / `top_scored_arm`",
         ("recommended_arm", "top_scored_arm")),
        ("what the card decomposes",
         ("attribution_source", "BLIP_TERM_LANGUAGE", "_why_not", "top_tailoring_variables")),
        ("standard errors and the covariance",
         ("cross_covariance", "sandwich_covariance", "contrast_standard_error",
          "blip_standard_error")),
        ("`linalg`",
         ("linalg", "weighted_least_squares", "gauss_jordan_inverse",
          "sandwich_product", "matmul", "cholesky_inverse")),
        ("the serving ensemble", ("SERVING_ENSEMBLE", "BayesianModelAverager")),
        ("the candidate set and the abstention rate",
         ("candidate_arms", "robustly_distinguishable", "simultaneous_alpha",
          "SANDWICH_INFLATION")),
        ("the arm and molecule vocabularies",
         ("arms.py", "formulary.py", "ARM_CANDIDATES", "normalize_arm")),
        ("Layer 1 ingestion and the contract",
         ("DataContractError", "PLAUSIBLE_RANGES", "FEATURE_DEFAULTS",
          "_has_adjuster", "OBSERVED_AS", "LeakageError")),
        ("policy value and off-policy evaluation",
         ("ipw_policy_value", "sequential_policy_value", "sequential_dr_value",
          "MIN_OPE_EFFECTIVE_SAMPLE")),
        ("the validation gate", ("ValidationLadder", "deployment_readiness", "live_data")),
        ("the blip basis", ("BLIP_BASIS", "blip_modifier", "das28_squared")),
        ("the safety layer", ("SafetyLayer", "FeasibleSet", "SafetyFlag", "arm_removed")),
        ("`cli audit` and the layer sections",
         ("cli audit", "audit_ingestion", "audit_decision", "audit_explanation",
          "audit_governance", "audit_safety")),
    )

    @classmethod
    def setUpClass(cls):
        cls.text = CLAUDE_MD.read_text()
        heads = sorted(
            (int(m.group(1)), m.start())
            for m in re.finditer(r"^(\d+)\. \*\*", cls.text, re.M)
        )
        end = cls.text.index("\n## What is real")
        cls.bodies = {
            n: cls.text[a:b]
            for (n, a), (_, b) in zip(heads, heads[1:] + [(0, end)])
        }
        cls.table = {
            label.strip(): [int(x) for x in numbers.split(",") if x.strip()]
            for label, numbers in re.findall(
                r"^\| (.+?) \| ([\d, ]+) \|$", cls.text, re.M
            )
        }

    @staticmethod
    def _matches(body, tokens):
        spans = re.findall(r"`([^`]+)`", body)
        for token in tokens:
            if token.startswith("cli "):
                if token in body:
                    return True
            elif any(token in span for span in spans):
                return True
        return False

    def test_every_row_lists_exactly_the_invariants_that_mention_it(self):
        for label, tokens in self.ROWS:
            derived = sorted(
                n for n, body in self.bodies.items() if self._matches(body, tokens)
            )
            with self.subTest(row=label):
                self.assertIn(label, self.table, "the index has lost this row")
                self.assertEqual(
                    self.table[label],
                    derived,
                    f"index row {label!r} is stale: CLAUDE.md says "
                    f"{self.table[label]}, the invariants say {derived}",
                )

    def test_the_coverage_claim_is_the_measured_one(self):
        """It says how much it reaches, because an index that implies
        completeness is worse than one that admits a gap."""
        covered = set()
        for _, tokens in self.ROWS:
            covered |= {
                n for n, body in self.bodies.items() if self._matches(body, tokens)
            }
        self.assertIn(
            f"**{len(covered)} of {len(self.bodies)}**",
            self.text,
            "the index's own coverage figure has gone stale",
        )

    def test_the_busiest_row_is_named_in_the_prose(self):
        """The argument for the index is that one quantity dominates; if that
        stops being true the framing above it is wrong."""
        counts = {
            label: sum(
                1 for body in self.bodies.values() if self._matches(body, tokens)
            )
            for label, tokens in self.ROWS
        }
        busiest = max(counts, key=counts.get)
        self.assertIn("q_values", busiest)
        self.assertGreaterEqual(counts[busiest], 10)


class TestCountTests(unittest.TestCase):
    """The count in `CLAUDE.md` is an instruction, so it has to be right.

    It is the first thing a reader checks a run against, and it goes stale every
    time a test is added without anyone thinking about the docs — which is the
    same drift that let the README disagree with `CLAUDE.md` on every headline
    figure.
    """

    def test_the_documented_test_count_matches_discovery(self):
        import unittest as _unittest

        discovered = _unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        actual = discovered.countTestCases()
        match = re.search(r"# (\d+) tests", CLAUDE_MD.read_text())
        self.assertIsNotNone(match, "CLAUDE.md no longer states a test count")
        self.assertEqual(
            int(match.group(1)),
            actual,
            f"CLAUDE.md says {match.group(1)} tests; discovery finds {actual}",
        )


class StaleFigureTests(unittest.TestCase):
    """The specific figures that went stale, pinned so they cannot come back.

    Not a general prose check — these are the exact strings that described the
    pre-covariance interval, and each one was live in at least one document or
    docstring until it was found by reading rather than by anything failing.
    """

    #: (stale figure, where an unqualified use of it would be wrong)
    SUPERSEDED = (
        "abstains on ~78%",
        "declines to name\none arm for **most** patients it sees — 78%",
    )

    def test_the_model_card_does_not_quote_the_pre_covariance_abstention(self):
        from treatmentrx.service import RecommendationService

        card = RecommendationService().model_card()
        self.assertNotIn("0.78,", repr(card))

    def test_the_model_card_and_the_docs_quote_the_same_abstention(self):
        """The motivating bug was two documents disagreeing, so assert agreement
        rather than a value.

        Pinning the literal here is what this test used to do, and it made the
        rate a number to remember in four places instead of one. The rate is a
        Monte Carlo output and moves whenever the decision rule does; what must
        never drift is the served card disagreeing with the prose beside it.
        """
        from treatmentrx.service import RecommendationService

        served = RecommendationService().model_card()["known_limitations"][
            "abstention"
        ]["pooled_rate"]
        self.assertIsInstance(served, float)
        for document in (CLAUDE_MD, README):
            quoted = re.search(r"abstains on ~(\d+)% of patients", document.read_text())
            with self.subTest(document=document.name):
                self.assertIsNotNone(
                    quoted, f"{document.name} no longer quotes an abstention rate"
                )
                self.assertEqual(
                    int(quoted.group(1)),
                    round(served * 100),
                    f"{document.name} and the served model card disagree",
                )

    def test_the_readme_quotes_the_current_abstention_rate(self):
        text = README.read_text()
        for stale in self.SUPERSEDED:
            with self.subTest(figure=stale):
                self.assertNotIn(stale, text)


if __name__ == "__main__":
    unittest.main()
