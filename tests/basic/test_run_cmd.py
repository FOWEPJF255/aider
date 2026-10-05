import os

import pytest  # noqa: F401

from aider.run_cmd import (KNOWN_SECRET_VALUES, PROVIDER_ENV_KEYS,
                           child_process_environ, register_known_secrets,
                           run_cmd)


@pytest.fixture(autouse=True)
def _clear_known_secrets():
    KNOWN_SECRET_VALUES.clear()
    yield
    KNOWN_SECRET_VALUES.clear()


def test_run_cmd_echo():
    command = "echo HelloWorld"
    exit_code, output = run_cmd(command)

    assert exit_code == 0
    assert "HelloWorld" in output


def test_child_process_environ_scrubs_provider_keys():
    """Issue #5658: child commands must not inherit provider credentials."""
    base = {
        "PATH": "/usr/bin",
        "OPENAI_API_KEY": "aider-provider-sentinel",
        "ANTHROPIC_API_KEY": "secret",
        "HOME": "/tmp",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "KEEP_ME": "yes",
    }
    scrubbed = child_process_environ(base=base)
    assert scrubbed["PATH"] == "/usr/bin"
    assert scrubbed["KEEP_ME"] == "yes"
    assert scrubbed["HOME"] == "/tmp"
    for key in PROVIDER_ENV_KEYS:
        assert key not in scrubbed
    # Extra overlays must win (e.g. GIT_EDITOR for /git).
    with_extra = child_process_environ(base=base, extra={"GIT_EDITOR": "true"})
    assert with_extra["GIT_EDITOR"] == "true"
    assert "OPENAI_API_KEY" not in with_extra


def test_child_process_environ_scrubs_by_secret_value():
    sentinel = "aider-provider-sentinel"
    register_known_secrets(sentinel)
    base = {
        "PATH": "/usr/bin",
        "CUSTOM_PROVIDER_SECRET": sentinel,
        "KEEP_ME": "yes",
    }
    scrubbed = child_process_environ(base=base)
    assert "CUSTOM_PROVIDER_SECRET" not in scrubbed
    assert scrubbed["KEEP_ME"] == "yes"


def test_child_process_environ_scrubs_pattern_suffix_keys():
    base = {
        "PATH": "/usr/bin",
        "DEEPSEEK_API_KEY": "ds-key",
        "MY_APP_TOKEN": "tok",
        "KEEP_ME": "yes",
    }
    scrubbed = child_process_environ(base=base)
    assert "DEEPSEEK_API_KEY" not in scrubbed
    assert "MY_APP_TOKEN" not in scrubbed
    assert scrubbed["KEEP_ME"] == "yes"


def test_child_process_environ_git_profile_keeps_git_vars():
    base = {
        "PATH": "/usr/bin",
        "GIT_CONFIG_GLOBAL": "/tmp/gitconfig",
        "SSH_AUTH_SOCK": "/tmp/ssh",
        "OPENAI_API_KEY": "secret",
    }
    scrubbed = child_process_environ(base=base, profile="git")
    assert scrubbed["GIT_CONFIG_GLOBAL"] == "/tmp/gitconfig"
    assert scrubbed["SSH_AUTH_SOCK"] == "/tmp/ssh"
    assert "OPENAI_API_KEY" not in scrubbed


def test_run_cmd_subprocess_does_not_expose_openai_key(monkeypatch):
    """Presence-only check: /run-style subprocess must not see OPENAI_API_KEY."""
    sentinel = "aider-provider-sentinel"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    register_known_secrets(sentinel)
    # Force subprocess path (no interactive pexpect).
    monkeypatch.setattr("aider.run_cmd.sys.stdin.isatty", lambda: False)
    if os.name == "nt":
        command = (
            "python -c \"import os,sys; sys.exit(0 if os.environ.get('OPENAI_API_KEY') else 1)\""
        )
    else:
        command = 'test -n "$OPENAI_API_KEY"'
    exit_code, _output = run_cmd(command)
    # Key scrubbed => Windows python exits 1; Unix test -n fails with 1.
    assert exit_code != 0


def test_run_cmd_subprocess_does_not_expose_renamed_secret_value(monkeypatch):
    sentinel = "aider-provider-sentinel"
    monkeypatch.setenv("RENAMED_SECRET", sentinel)
    register_known_secrets(sentinel)
    monkeypatch.setattr("aider.run_cmd.sys.stdin.isatty", lambda: False)
    if os.name == "nt":
        command = (
            "python -c \"import os,sys; sys.exit(0 if os.environ.get('RENAMED_SECRET') else 1)\""
        )
    else:
        command = 'test -n "$RENAMED_SECRET"'
    exit_code, _output = run_cmd(command)
    assert exit_code != 0
