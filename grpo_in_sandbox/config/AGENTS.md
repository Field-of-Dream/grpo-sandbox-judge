# TASK CONFIGURATIONS

**Location:** config/

## OVERVIEW

Runtime configurations for benchmark tasks.

## FILES

| File | Purpose |
|------|---------|
| general.yaml | General settings |
| runtime.example.yaml | Runtime config template |
| product_team.yaml | Product team config |
| company_team.yaml | Company team config |

## WHERE TO LOOK

Team configuration definitions in agent_configs.py:
- `load_team_from_yaml` loads product_team.yaml or company_team.yaml
- Used by AgentTeam class for multi-agent teams

## CONVENTIONS

- YAML config files for task parameters
- runtime.example.yaml is template (rename to runtime.yaml to use)
- product_team.yaml and company_team.yaml define agent team templates