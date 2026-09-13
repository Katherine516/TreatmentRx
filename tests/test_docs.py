"""The docs are a deliverable here, so a few of their claims are testable.

This repo's stated value is that a reader can tell a measured number from a
placeholder. That only holds while the prose agrees with the code, and it stopped
holding: `CLAUDE.md` was updated after the dWOLS cross-arm covariance was kept and
`README.md` was not, so the two documents disagreed on every headline table —
coverage 95% against 98%, abstention 67% against 78%, transfer 57-89% against
70-95% — and ten source docstrings still carried the pre-fix figures.

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
        rendered = repr(card)
        self.assertIn("0.67", rendered)
        self.assertNotIn("0.78,", rendered)

    def test_the_readme_quotes_the_current_abstention_rate(self):
        text = README.read_text()
        for stale in self.SUPERSEDED:
            with self.subTest(figure=stale):
                self.assertNotIn(stale, text)


if __name__ == "__main__":
    unittest.main()
