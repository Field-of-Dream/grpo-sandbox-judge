# TASK CONFIGURATIONS

**Location:** config/

## OVERVIEW
6 YAML files for prompt templates, runtime settings, and multi-agent team definitions. Loaded by config.py (AgentConfig dataclass) and cli.py (load_runtime_settings).

## FILES

| File | Purpose |
|------|---------|
| general.yaml | System/instance prompt templates + container paths |
| runtime.example.yaml | CLI defaults template (llm_name, llm_base_url, api_key) |
| product_team.yaml | Product team: ProductManager + Designer + Developer |
| company_team.yaml | Company team: TechLead + SeniorDev + JuniorDev |

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Prompt templates | general.yaml | Used by AgentArgs via config.py load_prompt_config |
| CLI defaults | runtime.example.yaml | Loaded by cli.py load_runtime_settings |
| Multi-agent teams | product_team.yaml, company_team.yaml | Loaded by agent_configs.py load_team_from_yaml |

## CONVENTIONS
- YAML configs define system_prompt, instance_prompt, working_dir, input_dir, output_dir
- {working_dir}, {input_dir}, {output_dir} placeholders auto-replaced with container paths
- Runtime settings support CLI flag > env var > YAML file > default precedence
- Agent teams: each agent has name, role, system_prompt, instance_prompt, description, capabilities