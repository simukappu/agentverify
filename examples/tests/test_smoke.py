"""Smoke tests for the examples directory structure and file existence.

Validates: Requirements 1.1, 1.5, 1.6, 1.7, 2.1, 2.5, 2.6, 2.7, 3.1, 3.5, 3.6, 5.1, 5.2, 5.4
"""

from pathlib import Path

import yaml

# Base path: examples/ directory (parent of tests/)
EXAMPLES_DIR = Path(__file__).parent.parent
# agentverify project root (parent of examples/)
PROJECT_ROOT = EXAMPLES_DIR.parent


class TestDirectoryStructure:
    """Verify all required files and directories exist."""

    def test_mcp_server_github_server(self):
        assert (EXAMPLES_DIR / "mcp-server" / "github_server.py").is_file()

    def test_mcp_server_readme(self):
        assert (EXAMPLES_DIR / "mcp-server" / "README.md").is_file()

    def test_strands_agent(self):
        assert (EXAMPLES_DIR / "strands-weather-forecaster" / "weather_agent.py").is_file()

    def test_strands_pyproject(self):
        """strands-weather-forecaster uses the built-in from_strands adapter;
        it doesn't ship its own pyproject.toml — the parent agentverify
        [all] extras provide strands-agents.
        """
        # No pyproject.toml expected for this example.
        pass

    def test_strands_conftest(self):
        assert (EXAMPLES_DIR / "strands-weather-forecaster" / "tests" / "conftest.py").is_file()

    def test_strands_test_file(self):
        assert (EXAMPLES_DIR / "strands-weather-forecaster" / "tests" / "test_weather_forecaster.py").is_file()

    def test_strands_cassette(self):
        assert (EXAMPLES_DIR / "strands-weather-forecaster" / "tests" / "cassettes" / "weather_seattle.yaml").is_file()

    def test_langchain_agent(self):
        assert (EXAMPLES_DIR / "langchain-issue-triage" / "agent.py").is_file()

    def test_langchain_pyproject(self):
        assert (EXAMPLES_DIR / "langchain-issue-triage" / "pyproject.toml").is_file()

    def test_langchain_readme(self):
        assert (EXAMPLES_DIR / "langchain-issue-triage" / "README.md").is_file()

    def test_langchain_conftest(self):
        assert (EXAMPLES_DIR / "langchain-issue-triage" / "tests" / "conftest.py").is_file()

    def test_langchain_test_file(self):
        assert (EXAMPLES_DIR / "langchain-issue-triage" / "tests" / "test_issue_triage.py").is_file()

    def test_langchain_cassette_mock(self):
        assert (EXAMPLES_DIR / "langchain-issue-triage" / "tests" / "cassettes" / "issue_triage_mock.yaml").is_file()

    def test_langchain_cassette_real(self):
        assert (EXAMPLES_DIR / "langchain-issue-triage" / "tests" / "cassettes" / "issue_triage_real.yaml").is_file()

    def test_langgraph_supervisor_agent(self):
        assert (EXAMPLES_DIR / "langgraph-multi-agent-supervisor" / "agent.py").is_file()

    def test_langgraph_supervisor_pyproject(self):
        assert (EXAMPLES_DIR / "langgraph-multi-agent-supervisor" / "pyproject.toml").is_file()

    def test_langgraph_supervisor_readme(self):
        assert (EXAMPLES_DIR / "langgraph-multi-agent-supervisor" / "README.md").is_file()

    def test_langgraph_supervisor_conftest(self):
        assert (EXAMPLES_DIR / "langgraph-multi-agent-supervisor" / "tests" / "conftest.py").is_file()

    def test_langgraph_supervisor_test_file(self):
        assert (EXAMPLES_DIR / "langgraph-multi-agent-supervisor" / "tests" / "test_supervisor.py").is_file()

    def test_langgraph_supervisor_cassette(self):
        assert (
            EXAMPLES_DIR
            / "langgraph-multi-agent-supervisor"
            / "tests"
            / "cassettes"
            / "supervisor_faang.yaml"
        ).is_file()

    def test_openai_agents_llm_as_a_judge_agent(self):
        assert (EXAMPLES_DIR / "openai-agents-llm-as-a-judge" / "agent.py").is_file()

    def test_openai_agents_llm_as_a_judge_pyproject(self):
        assert (EXAMPLES_DIR / "openai-agents-llm-as-a-judge" / "pyproject.toml").is_file()

    def test_openai_agents_llm_as_a_judge_readme(self):
        assert (EXAMPLES_DIR / "openai-agents-llm-as-a-judge" / "README.md").is_file()

    def test_openai_agents_llm_as_a_judge_conftest(self):
        assert (
            EXAMPLES_DIR / "openai-agents-llm-as-a-judge" / "tests" / "conftest.py"
        ).is_file()

    def test_openai_agents_llm_as_a_judge_test_file(self):
        assert (
            EXAMPLES_DIR
            / "openai-agents-llm-as-a-judge"
            / "tests"
            / "test_llm_as_a_judge.py"
        ).is_file()

    def test_openai_agents_llm_as_a_judge_cassette(self):
        assert (
            EXAMPLES_DIR
            / "openai-agents-llm-as-a-judge"
            / "tests"
            / "cassettes"
            / "detective_in_space.yaml"
        ).is_file()

    def test_custom_converter_python_agent_agent(self):
        assert (EXAMPLES_DIR / "custom-converter-python-agent" / "agent.py").is_file()

    def test_custom_converter_python_agent_converter(self):
        assert (EXAMPLES_DIR / "custom-converter-python-agent" / "converter.py").is_file()

    def test_custom_converter_python_agent_pyproject(self):
        assert (EXAMPLES_DIR / "custom-converter-python-agent" / "pyproject.toml").is_file()

    def test_custom_converter_python_agent_readme(self):
        assert (EXAMPLES_DIR / "custom-converter-python-agent" / "README.md").is_file()

    def test_custom_converter_python_agent_conftest(self):
        assert (
            EXAMPLES_DIR / "custom-converter-python-agent" / "tests" / "conftest.py"
        ).is_file()

    def test_custom_converter_python_agent_test_file(self):
        assert (
            EXAMPLES_DIR
            / "custom-converter-python-agent"
            / "tests"
            / "test_tax_agent.py"
        ).is_file()

    def test_custom_converter_python_agent_cassette(self):
        assert (
            EXAMPLES_DIR
            / "custom-converter-python-agent"
            / "tests"
            / "cassettes"
            / "tax_calculation.yaml"
        ).is_file()


class TestCassetteYamlValidity:
    """Verify cassette files are loadable as YAML with expected keys."""

    def test_strands_cassette_yaml(self):
        path = EXAMPLES_DIR / "strands-weather-forecaster" / "tests" / "cassettes" / "weather_seattle.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "metadata" in data
        assert "interactions" in data

    def test_langchain_mock_cassette_yaml(self):
        path = EXAMPLES_DIR / "langchain-issue-triage" / "tests" / "cassettes" / "issue_triage_mock.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "metadata" in data
        assert "interactions" in data

    def test_langchain_real_cassette_yaml(self):
        path = EXAMPLES_DIR / "langchain-issue-triage" / "tests" / "cassettes" / "issue_triage_real.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "metadata" in data
        assert "interactions" in data

    def test_langgraph_supervisor_cassette_yaml(self):
        path = (
            EXAMPLES_DIR
            / "langgraph-multi-agent-supervisor"
            / "tests"
            / "cassettes"
            / "supervisor_faang.yaml"
        )
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "metadata" in data
        assert "interactions" in data

    def test_openai_agents_llm_as_a_judge_cassette_yaml(self):
        path = (
            EXAMPLES_DIR
            / "openai-agents-llm-as-a-judge"
            / "tests"
            / "cassettes"
            / "detective_in_space.yaml"
        )
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "metadata" in data
        assert "interactions" in data

    def test_custom_converter_python_agent_cassette_yaml(self):
        path = (
            EXAMPLES_DIR
            / "custom-converter-python-agent"
            / "tests"
            / "cassettes"
            / "tax_calculation.yaml"
        )
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "metadata" in data
        assert "interactions" in data


class TestReadmeFiles:
    """Verify all README.md files exist."""

    def test_strands_readme_exists(self):
        # strands-weather-forecaster doesn't ship its own README — it's
        # documented entirely in the main README's "Real-World Examples"
        # section. Intentionally a no-op.
        pass

    def test_langchain_readme_exists(self):
        assert (EXAMPLES_DIR / "langchain-issue-triage" / "README.md").is_file()

    def test_langgraph_supervisor_readme_exists(self):
        assert (EXAMPLES_DIR / "langgraph-multi-agent-supervisor" / "README.md").is_file()

    def test_openai_agents_llm_as_a_judge_readme_exists(self):
        assert (EXAMPLES_DIR / "openai-agents-llm-as-a-judge" / "README.md").is_file()

    def test_custom_converter_python_agent_readme_exists(self):
        assert (EXAMPLES_DIR / "custom-converter-python-agent" / "README.md").is_file()

    def test_mcp_server_readme_exists(self):
        assert (EXAMPLES_DIR / "mcp-server" / "README.md").is_file()


class TestMainReadmeExamplesLink:
    """Verify the main README.md contains an examples reference link."""

    def test_main_readme_contains_examples_link(self):
        readme_path = PROJECT_ROOT / "README.md"
        assert readme_path.is_file(), "Main README.md not found"
        content = readme_path.read_text()
        assert "examples/" in content, "Main README.md does not contain 'examples/' reference"


class TestBuiltinAdapters:
    """Verify built-in framework adapters are importable."""

    def test_strands_adapter_importable(self):
        from agentverify.frameworks.strands import from_strands

        assert callable(from_strands)

    def test_langchain_adapter_importable(self):
        from agentverify.frameworks.langchain import from_langchain

        assert callable(from_langchain)

    def test_langgraph_adapter_importable(self):
        from agentverify.frameworks.langgraph import from_langgraph

        assert callable(from_langgraph)

    def test_openai_agents_adapter_importable(self):
        from agentverify.frameworks.openai_agents import from_openai_agents

        assert callable(from_openai_agents)
