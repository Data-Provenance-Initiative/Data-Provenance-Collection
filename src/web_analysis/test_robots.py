import unittest

from parse_robots import interpret_robots, parse_robots_txt


class TestRobotsTxtInterpretation(unittest.TestCase):

    def test_complete_block_all_agents(self):
        """Tests that all agents are completely blocked with a straightforward rule."""
        agent_rules = {"*": {"Disallow": ["/"]}}
        all_agents = ["GoogleBot", "BingBot", "CustomBot"]
        expected = {
            "*": "all",
            "GoogleBot": "all",
            "BingBot": "all",
            "CustomBot": "all",
        }
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Agents should be completely blocked, but weren't.",
        )

    def test_no_block_all_agents(self):
        """Tests that no agents are blocked when Disallow is explicitly empty."""
        agent_rules = {"*": {"Disallow": [""]}}
        all_agents = ["GoogleBot", "BingBot", "CustomBot"]
        expected = {
            "*": "none",
            "GoogleBot": "none",
            "BingBot": "none",
            "CustomBot": "none",
        }
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Agents should not be blocked at all, but they are.",
        )

    def test_partial_block_specific_directories(self):
        """Tests blocking of specific directories for all agents."""
        agent_rules = {"*": {"Disallow": ["/private", "/tmp", "/config"]}}
        all_agents = ["GoogleBot", "BingBot", "CustomBot"]
        expected = {
            "*": "some",
            "GoogleBot": "some",
            "BingBot": "some",
            "CustomBot": "some",
        }
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Agents should be partially blocked, but blocking level was incorrect.",
        )

    def test_agent_specific_allowance(self):
        """Tests specific allow rules that override general disallow rules for specific agents."""
        agent_rules = {
            "*": {"Disallow": ["/"]},
            "GoogleBot": {"Allow": ["/public"], "Disallow": ["/"]},
            "BingBot": {"Allow": ["/public/data"], "Disallow": ["/private"]},
        }
        all_agents = ["GoogleBot", "BingBot", "YahooBot"]
        expected = {
            "*": "all",
            "GoogleBot": "some",
            "BingBot": "some",
            "YahooBot": "all",
        }
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Specific allow rules should override general disallows, but didn't work as expected.",
        )

    def test_empty_and_missing_rules(self):
        """Tests the absence of any rules and empty rules handling."""
        agent_rules = {"*": {"Disallow": [], "Allow": []}, "BingBot": {}}
        all_agents = ["GoogleBot", "BingBot", "CustomBot"]
        expected = {
            "*": "none",
            "GoogleBot": "none",
            "BingBot": "none",
            "CustomBot": "none",
        }
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "No rules should mean no blocking, but the result was different.",
        )

    def test_malformed_rules_handling(self):
        """Tests handling of malformed rules, such as incorrect paths and wildcards."""
        agent_rules = {
            "*": {"Disallow": ["/*.php", "/?id=123"]},
            "GoogleBot": {"Disallow": ["*.html", "/test/*/index"]},
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Malformed rules should lead to some blocking, but interpretation was not as expected.",
        )

    def test_conflicting_rules(self):
        """Tests how conflicting rules are resolved within the same user agent."""
        agent_rules = {
            "GoogleBot": {"Disallow": ["/private"], "Allow": ["/private/stats"]}
        }
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Conflicting rules should result in some blocking for specific paths, but they didn't.",
        )

    def test_specific_vs_general_agent_rules(self):
        """Tests the precedence of specific user-agent rules over the general wildcard rules."""
        agent_rules = {
            "*": {"Disallow": ["/private"]},
            "GoogleBot": {"Disallow": [""]},
            "CustomBot": {"Disallow": ["/"]},
        }
        all_agents = ["GoogleBot", "BingBot", "CustomBot"]
        expected = {
            "*": "some",
            "GoogleBot": "none",
            "BingBot": "some",
            "CustomBot": "all",
        }
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Specific user-agent rules should take precedence over general rules, but didn't.",
        )

    def test_non_standard_paths(self):
        """Tests handling of non-standard paths that might be seen in real-world scenarios."""
        agent_rules = {"*": {"Disallow": ["/dev/*", "/staging/env"]}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Non-standard paths should be partially blocked, but weren't handled as expected.",
        )

    def test_complete_access_vs_explicit_disallow(self):
        """Tests explicit full access versus explicit disallow for different agents."""
        agent_rules = {"*": {"Disallow": ["/"]}, "GoogleBot": {"Disallow": []}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "all", "GoogleBot": "none", "BingBot": "all"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "GoogleBot should have no restrictions while others are fully blocked. There may be an issue with rule inheritance or interpretation.",
        )

    def test_multiple_agents_with_varying_rules(self):
        """Tests multiple agents with different sets of rules."""
        agent_rules = {
            "*": {"Disallow": ["/private"]},
            "GoogleBot": {"Disallow": ["/secret"], "Allow": ["/secret/info"]},
            "BingBot": {"Disallow": ["/private", "/confidential"]},
        }
        all_agents = ["GoogleBot", "BingBot", "YahooBot"]
        expected = {
            "*": "some",
            "GoogleBot": "some",
            "BingBot": "some",
            "YahooBot": "some",
        }
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "All agents should have varying block levels based on specific rules. Ensure rules are not being universally applied.",
        )

    def test_complex_allow_disallow_combinations(self):
        """Tests complex combinations of Allow and Disallow to ensure correct precedence and interpretation."""
        agent_rules = {
            "*": {"Disallow": ["/"], "Allow": ["/public", "/public_html"]},
            "GoogleBot": {"Allow": ["/private/stats"], "Disallow": ["/private"]},
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "The precedence of Allow/Disallow rules isn't respected. Check if Allow rules correctly override Disallow rules when expected.",
        )

    def test_overlapping_agent_rules(self):
        """Tests overlapping agent rules to see if specific agent rules take precedence over wildcard rules."""
        agent_rules = {
            "*": {"Disallow": ["/tmp"]},
            "GoogleBot": {"Disallow": ["/tmp", "/test"], "Allow": ["/tmp/readme"]},
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Specific rules for GoogleBot should properly overlap with general ones, ensuring GoogleBot's access is correctly interpreted.",
        )

    def test_disallow_empty_path(self):
        """Tests the interpretation of an empty path in Disallow which implies no restriction."""
        agent_rules = {"*": {"Disallow": [""]}, "GoogleBot": {"Disallow": []}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "none", "GoogleBot": "none", "BingBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "An empty Disallow should imply no restrictions, but the interpretation did not reflect this.",
        )

    def test_invalid_and_nonexistent_agents(self):
        """Tests how the function handles agents not defined in the rules and agents with invalid rules."""
        agent_rules = {"GoogleBot": {"Disallow": ["/api"], "Allow": ["/api/read"]}}
        all_agents = ["GoogleBot", "NonExistentBot"]
        expected = {"*": "none", "GoogleBot": "some", "NonExistentBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Non-existent agents should not have any rules applied, while existing ones should follow defined rules.",
        )

    def test_edge_cases_with_special_characters(self):
        """Tests edge cases with paths that include special characters and uncommon but valid configurations."""
        agent_rules = {"*": {"Disallow": ["/a*b", "/c?d", "/e(f)g"]}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Special characters in paths should be handled correctly, ensuring that wildcards and other symbols don't lead to unexpected blocking.",
        )

    def test_allow_all_except_one(self):
        """Tests allowing all paths except one specific path."""
        agent_rules = {"*": {"Disallow": ["/private"], "Allow": ["/"]}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "All paths should be allowed except the explicitly disallowed one. This test checks if exceptions are handled correctly.",
        )

    def test_explicit_exception(self):
        """Tests scenarios where specific paths are explicitly allowed in the midst of general disallowances."""
        agent_rules = {
            "*": {"Disallow": ["/"], "Allow": ["/public", "/public/api"]},
            "GoogleBot": {
                "Allow": ["/public/stats"],
                "Disallow": ["/private", "/private/data"],
            },
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Specific exceptions within a general disallowance should be respected, especially for specific paths allowed for GoogleBot.",
        )

    def test_detailed_directory_structure(self):
        """Tests deep nested directory structures with various permissions."""
        agent_rules = {
            "*": {
                "Disallow": ["/users", "/users/private/"],
                "Allow": ["/users/public"],
            },
            "GoogleBot": {
                "Disallow": ["/users/private/documents"],
                "Allow": ["/users/public/images"],
            },
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Complex directory structures with multiple layers of permissions should be interpreted correctly for each agent.",
        )

    def test_partial_path_allowances(self):
        """Tests partial path allowances that could potentially conflict with general rules."""
        agent_rules = {"GoogleBot": {"Disallow": ["/data"], "Allow": ["/data/public"]}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "none", "GoogleBot": "some", "BingBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Allowances on specific subpaths should override broader disallows for the same paths, but only for the specified agent.",
        )

    def test_multi_agent_specific_rules(self):
        """Tests complex multi-agent configurations with overlapping and conflicting rules."""
        agent_rules = {
            "*": {"Disallow": ["/confidential"]},
            "GoogleBot": {"Allow": ["/confidential/reports"]},
            "BingBot": {"Disallow": ["/confidential", "/confidential/reports"]},
        }
        all_agents = ["GoogleBot", "BingBot", "YahooBot"]
        expected = {
            "*": "some",
            "GoogleBot": "none",
            "BingBot": "some",
            "YahooBot": "some",
        }
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "The specific allowances and disallows for multiple agents should be accurately reflected, showing proper precedence and conflict resolution.",
        )

    # def test_mixed_case_sensitivity(self):
    #     """Tests handling of case sensitivity in agent names and path rules."""
    #     agent_rules = {
    #         "googlebot": {"Disallow": ["/Private"]},
    #         "GoogleBot": {"Allow": ["/private/stats"]},
    #     }
    #     all_agents = ["googlebot", "GoogleBot"]
    #     expected = {"*": "none", "googlebot": "some", "GoogleBot": "some"}
    #     self.assertEqual(
    #         interpret_robots(agent_rules, all_agents),
    #         expected,
    #         "Case sensitivity in agent names and paths should be managed uniformly, ensuring consistent rule application regardless of case variations.",
    #     )

    def test_special_characters_in_paths(self):
        """Tests paths containing special characters like space, ampersand, etc."""
        agent_rules = {
            "*": {"Disallow": ["/test path", "/data&info", "/special#char?"]}
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Special characters in paths should be interpreted correctly, ensuring paths with spaces, symbols, and punctuation are processed accurately.",
        )

    def test_very_specific_rules(self):
        """Tests extremely specific disallow and allow combinations."""
        agent_rules = {
            "*": {"Disallow": ["/temp/*"], "Allow": ["/temp/readme.txt"]},
            "GoogleBot": {
                "Disallow": ["/temp/*"],
                "Allow": ["/temp/readme.txt", "/temp/images"],
            },
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Very specific rules should be correctly prioritized, allowing exceptions within broader disallowances for designated paths.",
        )

    def test_complete_block_except_certain_file_types(self):
        """Tests that all paths are blocked except specific file types."""
        agent_rules = {"*": {"Disallow": ["/"], "Allow": ["/images/", "/styles/"]}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "File types should be accessible, but broader disallowance isn't overridden as expected.",
        )

    def test_implicit_allow(self):
        """Tests the handling of implicit allow when no explicit rules are set."""
        agent_rules = {}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Agents should implicitly have complete access if no rules are set.",
        )

    def test_path_depth_handling(self):
        """Tests rule application accuracy beyond specified path depth."""
        agent_rules = {"*": {"Disallow": ["/dir/"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Disallowed directory should extend to all subdirectories and files, but it doesn't.",
        )

    def test_case_sensitivity_of_paths(self):
        """Tests the case sensitivity of path rules."""
        agent_rules = {"*": {"Disallow": ["/Dir/"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Path case sensitivity isn't handled properly, leading to incorrect access rules.",
        )

    def test_complex_query_parameters(self):
        """Tests correct handling of URLs with complex query parameters."""
        agent_rules = {"*": {"Disallow": ["/search", "/search?query=allowed"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Query parameters should be respected in disallow rules, but aren't being parsed correctly.",
        )

    def test_overlap_between_allow_and_disallow(self):
        """Tests the rule precedence when allow and disallow paths overlap."""
        agent_rules = {
            "*": {"Disallow": ["/directory/"], "Allow": ["/directory/subdirectory/"]}
        }
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Allow should override Disallow for overlapping paths, but it doesn't.",
        )

    def test_rule_priority_across_different_user_agents(self):
        """Ensures specific user-agent rules take precedence over generic ones."""
        agent_rules = {
            "*": {"Disallow": ["/private/"]},
            "GoogleBot": {"Allow": ["/private/stats"]},
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "none", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "GoogleBot specific rules should override general rules, but the priority isn't maintained.",
        )

    def test_allow_specific_files_in_disallowed_directory(self):
        """Tests that specific files within a generally disallowed directory are accessible."""
        agent_rules = {
            "*": {"Disallow": ["/private/"], "Allow": ["/private/index.html"]}
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Specific files in a disallowed directory should be accessible, but are not.",
        )

    def test_disallow_specific_file_types(self):
        """Tests blocking specific file types within otherwise accessible directories."""
        agent_rules = {
            "*": {"Allow": ["/downloads/"], "Disallow": ["/downloads/*.exe"]}
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Specific file types should be blocked, but are accessible.",
        )

    def test_redirect_handling(self):
        """Tests how redirects within disallowed paths are handled."""
        agent_rules = {"*": {"Disallow": ["/redirect/"]}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Redirects within disallowed paths should be blocked, but aren't being handled correctly.",
        )

    def test_url_casing_impact(self):
        """Tests the impact of URL case sensitivity on rule application."""
        agent_rules = {"*": {"Disallow": ["/CaseSensitivePath/"]}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "URL case sensitivity is not handled correctly, leading to incorrect access controls.",
        )

    def test_disallow_with_wildcards(self):
        """Tests the correct handling of wildcards in Disallow directives."""
        agent_rules = {"*": {"Disallow": ["/images/*.jpg"]}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Wildcards in Disallow directives should restrict access to matching files, but do not.",
        )

    def test_robots_txt_without_user_agent_specification(self):
        """Tests behavior when no User-agent is specified and general rules apply."""
        agent_rules = {"*": {"Disallow": ["/"]}}
        all_agents = []
        expected = {"*": "all"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Rules without specific User-agent should apply to all robots, but they don't.",
        )

    def test_robots_txt_with_multiple_allow_and_disallow(self):
        """Tests multiple Allow and Disallow rules affecting the same paths."""
        agent_rules = {
            "GoogleBot": {"Allow": ["/folder/subfolder/"], "Disallow": ["/folder/"]}
        }
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Allow rules should override Disallow rules for the same path, but precedence is not maintained.",
        )

    def test_no_robots_txt(self):
        """Tests the behavior when no robots.txt file is present."""
        agent_rules = {}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "none", "GoogleBot": "none", "BingBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "In the absence of a robots.txt, all paths should be accessible, but they're not.",
        )

    def test_rules_application_to_specific_user_agents_only(self):
        """Tests that rules apply only to specified user agents and not to others."""
        agent_rules = {
            "GoogleBot": {"Disallow": ["/private/"]},
            "BingBot": {"Allow": ["/"]},
        }
        all_agents = ["GoogleBot", "BingBot", "YahooBot"]
        expected = {
            "*": "none",
            "GoogleBot": "some",
            "BingBot": "none",
            "YahooBot": "none",
        }
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Rules should apply specifically to defined user agents only, but they are affecting others.",
        )

    def test_allow_and_disallow_same_path(self):
        """Tests the scenario where the same path is both allowed and disallowed."""
        agent_rules = {"*": {"Allow": ["/path"], "Disallow": ["/path"]}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "none", "GoogleBot": "none", "BingBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "When the same path is both allowed and disallowed, the specific rule should take precedence, but it doesn't.",
        )

    def test_complex_wildcard_handling(self):
        """Tests complex wildcard scenarios in Disallow directives."""
        agent_rules = {"*": {"Disallow": ["/images/*.jpg", "/docs/*202?/"]}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "some", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Wildcards in paths should be handled correctly, blocking all matching files.",
        )

    def test_case_sensitivity_in_paths(self):
        """Tests the case sensitivity of paths in Disallow and Allow directives."""
        agent_rules = {"*": {"Disallow": ["/Data"], "Allow": ["/data/public"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Path case sensitivity should be correctly handled, distinguishing between '/Data' and '/data'.",
        )

    def test_absence_of_robots_txt(self):
        """Tests the behavior when no robots.txt is present or it's empty."""
        agent_rules = {}  # Simulating an empty or missing robots.txt
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "In the absence of a robots.txt, all paths should be accessible.",
        )

    def test_allow_disallow_same_path(self):
        """Tests the precedence of Allow over Disallow on the same path."""
        agent_rules = {"*": {"Disallow": ["/folder"], "Allow": ["/folder/subfolder"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Allow should take precedence over Disallow when both target the same path hierarchy.",
        )

    def test_wildcard_handling(self):
        """Tests the correct interpretation of wildcard characters in paths."""
        agent_rules = {"*": {"Disallow": ["/tmp/*"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Wildcards should block access to all subdirectories under /tmp, but they do not.",
        )

    def test_mixed_case_user_agents(self):
        """Tests whether the function correctly handles mixed-case user-agent names."""
        agent_rules = {
            "Googlebot": {"Disallow": ["/private"]},
            "googleBot": {"Allow": ["/private/stats"]},
        }
        all_agents = ["Googlebot"]
        expected = {"*": "none", "Googlebot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "User-agent names should be treated case-insensitively, but they are not.",
        )

    def test_empty_robots_txt(self):
        """Tests behavior when the robots.txt is empty, implying all access should be allowed."""
        agent_rules = {}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "An empty robots.txt should allow all access, but it does not.",
        )

    def test_precedence_of_directives(self):
        """Tests precedence of Allow over Disallow directives within the same robots.txt."""
        agent_rules = {"GoogleBot": {"Disallow": ["/"], "Allow": ["/public"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Allow directives should override conflicting Disallow directives, but they do not.",
        )

    def test_wildcard_handling(self):
        """Tests the correct interpretation of wildcard characters in paths."""
        agent_rules = {"*": {"Disallow": ["/tmp/*"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Wildcards should block access to all subdirectories under /tmp, but they do not.",
        )

    # def test_mixed_case_user_agents(self):
    #     """Tests whether the function correctly handles mixed-case user-agent names."""
    #     agent_rules = {
    #         "Googlebot": {"Disallow": ["/private"]},
    #         "googleBot": {"Allow": ["/private/stats"]},
    #     }
    #     all_agents = ["Googlebot"]
    #     expected = {"*": "none", "Googlebot": "some"}
    #     self.assertEqual(
    #         interpret_robots(agent_rules, all_agents),
    #         expected,
    #         "User-agent names should be treated case-insensitively, but they are not.",
    #     )

    # def test_query_string_sensitive_blocking(self):
    #     """Tests handling of URLs that include query strings in their disallow rules."""
    #     agent_rules = {"*": {"Disallow": ["/search?query=*"]}}
    #     all_agents = ["GoogleBot"]
    #     expected = {"*": "some", "GoogleBot": "some"}
    #     self.assertEqual(
    #         interpret_robots(agent_rules, all_agents),
    #         expected,
    #         "Disallowed paths with query strings should block matching URLs, but they do not.",
    #     )

    def test_empty_robots_txt(self):
        """Tests behavior when the robots.txt is empty, implying all access should be allowed."""
        agent_rules = {}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "An empty robots.txt should allow all access, but it does not.",
        )

    def test_precedence_of_directives(self):
        """Tests precedence of Allow over Disallow directives within the same robots.txt."""
        agent_rules = {"GoogleBot": {"Disallow": ["/"], "Allow": ["/public"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Allow directives should override conflicting Disallow directives, but they do not.",
        )

    def test_disallow_all_but_one(self):
        """Tests disallowing all paths except one specific allowed path."""
        agent_rules = {"*": {"Disallow": ["/"], "Allow": ["/index.html"]}}
        all_agents = ["GoogleBot", "BingBot"]
        expected = {
            "*": "some",  # Since there's an exception allowing '/index.html'
            "GoogleBot": "some",  # Access to '/index.html' allowed, others disallowed
            "BingBot": "some",  # Same as GoogleBot
        }
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Disallowing all except one file should block all but that file. Check for incorrect rule application.",
        )

    def test_case_insensitive_paths(self):
        """Tests that the path directives are case insensitive."""
        agent_rules = {"*": {"Disallow": ["/Folder"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Path directives should be case insensitive, ensuring '/Folder' matches '/folder'.",
        )

    def test_query_parameters_in_paths(self):
        """Tests that paths including query parameters are handled correctly."""
        agent_rules = {"*": {"Disallow": ["/search?query=public"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Paths with query parameters should be respected in disallow rules.",
        )

    def test_utf8_path_encodings(self):
        """Tests the handling of UTF-8 encoded paths."""
        agent_rules = {"*": {"Disallow": ["/résumé"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "UTF-8 paths should be correctly interpreted and matched.",
        )

    def test_empty_user_agent_disallow(self):
        """Tests behavior when Disallow is set but no user-agent is specified."""
        agent_rules = {"": {"Disallow": ["/private"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Rules without a user-agent should ideally not be processed.",
        )

    def test_multiple_allow_disallow_single_agent(self):
        """Tests multiple allow and disallow entries for a single agent to check conflict resolution."""
        agent_rules = {
            "GoogleBot": {
                "Disallow": ["/private", "/tmp"],
                "Allow": ["/private/stats", "/tmp/images"],
            }
        }
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "GoogleBot should have specific paths allowed despite broader disallows.",
        )

    def test_inheritance_of_general_disallow(self):
        """Tests if unspecified agents inherit general disallow rules."""
        agent_rules = {"*": {"Disallow": ["/"]}}
        all_agents = ["UnknownBot"]
        expected = {"*": "all", "UnknownBot": "all"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Unspecified agents should inherit general disallow rules.",
        )

    def test_specific_blocking_in_allowed_directory(self):
        """Tests specific disallow inside a generally allowed directory."""
        agent_rules = {"*": {"Allow": ["/data"], "Disallow": ["/data/private"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Access to '/data/private' should be blocked despite '/data' being allowed.",
        )

    # def test_case_insensitivity_in_user_agent(self):
    #     """Ensures user-agent matching is case insensitive."""
    #     agent_rules = {"googlebot": {"Disallow": ["/private"]}}
    #     all_agents = ["GoogleBot"]
    #     expected = {"*": "none", "GoogleBot": "some"}
    #     self.assertEqual(
    #         interpret_robots(agent_rules, all_agents),
    #         expected,
    #         "User-agent matching should be case insensitive.",
    #     )

    def test_wildcards_in_middle_of_path(self):
        """Tests correct handling of wildcards in the middle of paths."""
        agent_rules = {"*": {"Disallow": ["/data/*/private"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Wildcards in the middle of paths should correctly block access to matching directories.",
        )

    def test_specific_allow_over_general_disallow(self):
        """Tests a specific allow rule in a generally disallowed path."""
        agent_rules = {"*": {"Disallow": ["/data"], "Allow": ["/data/public"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Specific allow rules should override general disallow rules within the same path hierarchy.",
        )

    def test_file_type_exclusion(self):
        """Tests blocking specific file types in otherwise accessible directories."""
        agent_rules = {"*": {"Disallow": ["/images/*.jpg"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "JPG images in the '/images' directory should be blocked.",
        )

    def test_allow_overrides_disallow(self):
        """Tests that the Allow directive properly overrides a conflicting Disallow for the same path."""
        agent_rules = {"*": {"Disallow": ["/folder"], "Allow": ["/folder/subfolder"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Allow directive should override conflicting Disallow for the same path, ensuring access to /folder/subfolder.",
        )

    def test_empty_user_agent_and_path(self):
        """Tests behavior when user-agent and path are explicitly empty, which should imply no restrictions."""
        agent_rules = {"": {"Disallow": [""]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "An empty user-agent and path in Disallow should lead to no restrictions.",
        )

    def test_unicode_path_handling(self):
        """Tests the function's ability to handle paths with Unicode characters."""
        agent_rules = {"*": {"Disallow": ["/über"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Paths with Unicode characters should be properly disallowed.",
        )

    def test_multiple_disallow_with_wildcards(self):
        """Tests handling of multiple Disallow entries with wildcards for a single agent."""
        agent_rules = {"GoogleBot": {"Disallow": ["/tmp/*", "/private/*/config"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Multiple Disallow entries with wildcards should be correctly handled for GoogleBot.",
        )

    def test_conflicting_allow_disallow_same_path(self):
        """Tests resolution of conflicting Allow and Disallow rules for the exact same path."""
        agent_rules = {"*": {"Disallow": ["/data"], "Allow": ["/data"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Conflicting Allow and Disallow for the same path should default to allowing access.",
        )

    def test_specific_vs_generic_user_agents(self):
        """Tests specific user-agent rules taking precedence over generic wildcard rules."""
        agent_rules = {
            "*": {"Disallow": ["/private"]},
            "GoogleBot": {"Allow": ["/private/stats"]},
        }
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Specific user-agent rules should take precedence, allowing GoogleBot access to /private/stats despite a general disallow.",
        )

    def test_improperly_formatted_paths(self):
        """Tests handling of improperly formatted paths such as those with multiple slashes or trailing spaces."""
        agent_rules = {"*": {"Disallow": ["/bad//path/", "/trailing "]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Improperly formatted paths should be normalized and matched correctly.",
        )

    def test_allow_all_access(self):
        """Tests that absence of Disallow and presence of an Allow for all paths results in no restrictions."""
        agent_rules = {"*": {"Allow": ["/"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Allow for all paths should result in no restrictions for any agents.",
        )

    def test_multiple_user_agents_with_overlap_rules(self):
        """Tests handling of multiple user agents where some rules overlap and others contradict."""
        agent_rules = {
            "GoogleBot": {"Disallow": ["/private"], "Allow": ["/private/stats"]},
            "BingBot": {"Disallow": ["/private", "/public"]},
            "*": {"Allow": ["/public"]},
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "none", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Overlapping and contradicting rules across multiple user agents should be resolved correctly.",
        )

    def test_deeply_nested_rules(self):
        """Tests deeply nested and complex rule sets for proper resolution."""
        agent_rules = {
            "*": {"Disallow": ["/data"], "Allow": ["/data/public"]},
            "GoogleBot": {
                "Allow": ["/data/private/stats"],
                "Disallow": ["/data/private"],
            },
        }
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Deeply nested and complex rules should be interpreted correctly for all paths.",
        )

    def test_robustness_against_malformed_entries(self):
        """Tests the function's resilience against malformed entries."""
        agent_rules = {
            "GoogleBot": {
                "Disallow": [""]
            },  # Empty disallow should mean nothing is disallowed
            "*": {
                "Disallow": ["/validpath"],
                "Crawl-delay": ["10"],
            },  # Crawl-delay should be ignored
        }
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Malformed entries should be handled gracefully, ignoring invalid directives and processing valid ones.",
        )

    def test_performance_on_large_robots_txt(self):
        """Simulates a performance test on a very large robots.txt configuration."""
        agent_rules = {"*": {"Disallow": ["/" + str(i) for i in range(1000)]}}
        all_agents = ["GoogleBot"]
        # Expected to block GoogleBot from 1000 directories efficiently
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Performance test: Function should handle large rulesets efficiently.",
        )

    def test_conflicting_rules_across_agents(self):
        """Tests resolution of conflicting rules across different user agents."""
        agent_rules = {
            "GoogleBot": {"Disallow": ["/content"], "Allow": ["/content/public"]},
            "BingBot": {"Disallow": ["/content/public"]},
            "*": {"Allow": ["/content"]},
        }
        all_agents = ["GoogleBot", "BingBot"]
        expected = {"*": "none", "GoogleBot": "some", "BingBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Conflicting rules across agents should be resolved, with specific rules taking precedence.",
        )

    def test_similar_paths_distinction(self):
        """Ensures that similar paths are distinguished and not over-blocked."""
        agent_rules = {
            "*": {"Disallow": ["/data/reports"], "Allow": ["/data/reports/summary"]}
        }
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Disallow should not block '/data/reports/summary' when '/data/reports' is disallowed.",
        )

    def test_query_parameters_handling(self):
        """Tests that rules with query parameters are ignored as per standard but handled if specified."""
        agent_rules = {"*": {"Disallow": ["/search?query=private"]}}
        all_agents = ["GoogleBot"]
        expected = {"*": "none", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Paths with query parameters should ideally be ignored, ensuring they don't block unintended URLs.",
        )

    def test_whitespace_in_paths_handling(self):
        """Checks handling of paths that contain whitespace or are empty."""
        agent_rules = {
            "*": {"Disallow": [" /admin", "/register "]},
            "GoogleBot": {"Disallow": [""]},
        }
        all_agents = ["GoogleBot"]
        expected = {"*": "some", "GoogleBot": "none"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Whitespace in paths should be trimmed, and empty paths should imply no disallowance.",
        )

    def test_group_multiple_user_agents(self):
        """Tests that multiple consecutive user-agent lines form a group sharing the same rules.

        As per RFC 9309 Section 2.1 and Google's documentation, multiple user-agent lines
        that are not separated by blank lines should be treated as a group, with all
        subsequent rules applying to all agents in that group.
        """
        robots_txt = """
            User-agent: anthropic-ai
            User-agent: Google-Extended
            User-agent: GPTBot
            Disallow: /
            """
        agent_rules = parse_robots_txt(robots_txt)
        all_agents = ["anthropic-ai", "Google-Extended", "GPTBot"]
        expected = {
            "*": "none",
            "anthropic-ai": "all",
            "Google-Extended": "all",
            "GPTBot": "all",
        }
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Multiple stacked user-agents should each get the same rules applied.",
        )

    def test_group_separation_by_blank_line(self):
        """Tests that blank lines properly separate user-agent groups.

        According to Google's documentation on grouping rules, blank lines should separate
        different groups of rules. Non-standard directives like Crawl-delay should be ignored
        and should not affect the grouping of rules.
        """
        robots_txt = """
            User-agent: GPTBot
            Crawl-delay: 5

            User-agent: CCBot
            Disallow: /
            """
        agent_rules = parse_robots_txt(robots_txt)
        all_agents = ["GPTBot", "CCBot"]
        expected = {"*": "none", "GPTBot": "all", "CCBot": "all"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Crawl-delay should be ignored and repeated agents should work correctly.",
        )

    def test_case_insensitive_directives(self):
        """Tests that user-agent directive is case-insensitive.

        As per Google's documentation on user-agent line syntax, the "user-agent:" directive
        itself should be recognized regardless of case (e.g., "User-Agent:", "USER-AGENT:",
        or "user-agent:" are all valid).
        """
        robots_txt = """
            user-agent: GPTBot
            Disallow: /
            """
        agent_rules = parse_robots_txt(robots_txt)
        all_agents = ["GPTBot"]
        expected = {"*": "none", "GPTBot": "all"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "User-agent matching should be case-insensitive.",
        )

    def test_case_insensitive_agent_names(self):
        """Tests that user-agent names are matched case-insensitively.

        Different capitalizations of the same user-agent name (e.g., "Googlebot" vs "GoogleBot")
        should be treated as the same agent, with their rules being combined according to
        standard precedence rules.
        """
        robots_txt = """
            User-agent: Googlebot
            Disallow: /

            User-agent: GoogleBot
            Allow: /stats
            """
        agent_rules = parse_robots_txt(robots_txt)
        all_agents = ["Googlebot"]
        expected = {"*": "none", "Googlebot": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Different cases of Googlebot should be unified.",
        )

    def test_comment_handling(self):
        """Tests proper handling of comments in robots.txt.

        According to Google's documentation on syntax, comments start with # and continue
        to the end of the line. Comments should be completely ignored and should not affect
        the interpretation of subsequent directives.
        """
        robots_txt = """
            User-agent: GPTBot
            # Block GPTBot
            Disallow: /
            """
        agent_rules = parse_robots_txt(robots_txt)
        all_agents = ["GPTBot"]
        expected = {"*": "none", "GPTBot": "all"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Comments should not affect rule interpretation.",
        )

    def test_merge_same_agent_groups(self):
        """
        Tests that multiple groups for the same user agent are merged.

        As shown in Google's documentation, if there are multiple groups in robots.txt
        that are relevant to a specific user agent, Google's crawlers internally merge
        the groups.
        """
        robots_txt = """
            user-agent: googleBot-news
            disallow: /fish
            
            user-agent: *
            disallow: /carrots
            
            user-agent: googlebot-news
            disallow: /shrimp
            """
        agent_rules = parse_robots_txt(robots_txt)
        all_agents = ["googlebot-news"]
        expected = {"*": "some", "googlebot-news": "some"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Multiple groups for googlebot-news should be merged into one set of rules.",
        )

    def test_groups_with_non_standard_directives(self):
        """
        Tests that groups separated by non-standard directives are treated as one group.
        """
        robots_txt = """
        user-agent: a
        sitemap: https://example.com/sitemap.xml
        
        user-agent: b
        disallow: /
        """
        agent_rules = parse_robots_txt(robots_txt)

        # Test that both user agents are affected by the disallow rule
        all_agents = ["a", "b"]
        expected = {"*": "none", "a": "all", "b": "all"}
        self.assertEqual(
            interpret_robots(agent_rules, all_agents),
            expected,
            "Both user agents should be affected by the disallow rule despite sitemap directive.",
        )


if __name__ == "__main__":
    unittest.main()
