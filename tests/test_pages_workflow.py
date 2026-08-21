import re
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode


def _plain_yaml(node: Node) -> object:
    if isinstance(node, MappingNode):
        result: dict[str, object] = {}
        for key_node, value_node in node.value:
            key = _plain_yaml(key_node)
            assert isinstance(key, str)
            result[key] = _plain_yaml(value_node)
        return result
    if isinstance(node, SequenceNode):
        return [_plain_yaml(item) for item in node.value]
    assert isinstance(node, ScalarNode)
    return None if node.tag.endswith(":null") else node.value


def test_pages_workflow_builds_and_deploys_current_docs() -> None:
    workflow_path = Path(".github/workflows/pages.yml")
    workflow_text = workflow_path.read_text(encoding="utf-8")
    document = yaml.compose(workflow_text)
    assert document is not None
    workflow = _plain_yaml(document)
    assert isinstance(workflow, dict)

    triggers = workflow["on"]
    assert isinstance(triggers, dict)
    assert triggers["push"] == {
        "branches": ["main", "feat/**"],
        "tags": ["v*"],
    }
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"] == {}
    assert workflow["concurrency"] == {
        "group": "pages",
        "cancel-in-progress": "false",
    }

    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    build = jobs["build"]
    assert isinstance(build, dict)
    assert "if" not in build
    assert build["permissions"] == {"contents": "read", "pages": "read"}
    build_steps = build["steps"]
    assert isinstance(build_steps, list)
    checkout_steps = [
        step
        for step in build_steps
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert len(checkout_steps) == 1
    assert checkout_steps[0]["with"] == {"fetch-depth": "0"}
    build_commands = [step["run"] for step in build_steps if "run" in step]
    assert build_commands == [
        "uv sync --locked --all-groups",
        "uv run sphinx-polyversion --sequential docs/poly.py docs/_build/html",
    ]

    upload_steps = [
        step
        for step in build_steps
        if str(step.get("uses", "")).startswith("actions/upload-pages-artifact@")
    ]
    assert len(upload_steps) == 1
    assert upload_steps[0]["with"] == {"path": "docs/_build/html"}

    deploy = jobs["deploy"]
    assert isinstance(deploy, dict)
    assert deploy["if"] == "github.ref == 'refs/heads/main'"
    assert deploy["needs"] == "build"
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert deploy["environment"] == {
        "name": "github-pages",
        "url": "${{ steps.deployment.outputs.page_url }}",
    }
    deploy_steps = deploy["steps"]
    assert isinstance(deploy_steps, list)
    assert len(deploy_steps) == 1
    assert deploy_steps[0]["id"] == "deployment"
    assert str(deploy_steps[0]["uses"]).startswith("actions/deploy-pages@")

    action_lines = [line for line in workflow_text.splitlines() if "uses:" in line]
    assert action_lines
    for line in action_lines:
        assert re.search(r"uses:\s+[^@\s]+@[0-9a-f]{40}\s+# v\d+(?:\.\d+)*$", line)
