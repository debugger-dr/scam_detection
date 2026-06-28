import unittest

from mcp_tool_firewall.policy import PolicyEngine, inspect_tool_output


POLICY = {
    "default_action": "require_approval",
    "global_deny_patterns": [".env", "curl ", "| sh"],
    "tools": {
        "demo.weather": {"action": "allow"},
        "demo.shell_run": {"action": "block"},
    },
}


class PolicyEngineTests(unittest.TestCase):
    def test_allows_known_safe_tool(self) -> None:
        decision = PolicyEngine(POLICY).evaluate("demo.weather", {"city": "Lahore"})

        self.assertEqual(decision.action, "allow")

    def test_blocks_dangerous_argument_pattern_before_tool_policy(self) -> None:
        decision = PolicyEngine(POLICY).evaluate("demo.weather", {"city": ".env"})

        self.assertEqual(decision.action, "block")
        self.assertIn(".env", decision.matched_patterns)

    def test_blocks_shell_tool(self) -> None:
        decision = PolicyEngine(POLICY).evaluate("demo.shell_run", {"cmd": "echo hello"})

        self.assertEqual(decision.action, "block")

    def test_unknown_tools_require_approval(self) -> None:
        decision = PolicyEngine(POLICY).evaluate("unknown.tool", {})

        self.assertEqual(decision.action, "require_approval")

    def test_output_inspection_blocks_secret_leak(self) -> None:
        decision = inspect_tool_output(
            {"content": [{"type": "text", "text": "API_KEY=demo-secret-value"}]}
        )

        self.assertEqual(decision.action, "block")


if __name__ == "__main__":
    unittest.main()
