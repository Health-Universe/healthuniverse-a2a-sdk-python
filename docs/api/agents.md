# Agents API Reference

## StreamingAgent

::: health_universe_a2a.StreamingAgent
    options:
      show_root_heading: true
      members:
        - name
        - description
        - skills
        - stream
        - validate

## AsyncAgent

::: health_universe_a2a.AsyncAgent
    options:
      show_root_heading: true
      members:
        - name
        - description
        - skills
        - run_background
        - validate

## A2AAgentBase

::: health_universe_a2a.A2AAgentBase
    options:
      show_root_heading: true
      members:
        - name
        - description
        - skills
        - version
        - provider
        - extensions
        - execute
        - call_agent
