"""Multi-agent configuration system for defining AI agent teams with prompt templates."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

from .agent import Agent, AgentArgs

logger = logging.getLogger(__name__)


@dataclass
class AgentProfile:
    """Individual agent profile with prompt templates.

    Attributes:
        name: Agent identifier (e.g., "Dev1")
        role: Role in the company (e.g., "developer")
        system_prompt: System-level prompt defining agent behavior
        instance_prompt: Instance-level prompt template with {placeholder} support
        description: Optional human-readable description
        capabilities: List of capabilities this agent has
    """
    name: str
    role: str
    system_prompt: str
    instance_prompt: str = "{problem_statement}"
    description: str = ""
    capabilities: List[str] = field(default_factory=list)

    def format_system_prompt(self, **kwargs) -> str:
        return self.system_prompt.format(**kwargs)

    def format_instance_prompt(self, **kwargs) -> str:
        return self.instance_prompt.format(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "system_prompt": self.system_prompt,
            "instance_prompt": self.instance_prompt,
            "description": self.description,
            "capabilities": self.capabilities,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentProfile":
        return cls(
            name=data["name"],
            role=data["role"],
            system_prompt=data["system_prompt"],
            instance_prompt=data.get("instance_prompt", "{problem_statement}"),
            description=data.get("description", ""),
            capabilities=data.get("capabilities", []),
        )


class AgentTeam:
    """Team of agents managing multiple agent profiles."""

    def __init__(self, agents: Optional[List[AgentProfile]] = None):
        self.agents: List[AgentProfile] = agents or []
        self._by_name: Dict[str, AgentProfile] = {}
        self._by_role: Dict[str, List[AgentProfile]] = {}

        for agent in self.agents:
            self._add_agent(agent)

    def _add_agent(self, agent: AgentProfile):
        self._by_name[agent.name] = agent
        if agent.role not in self._by_role:
            self._by_role[agent.role] = []
        self._by_role[agent.role].append(agent)

    def add_agent(self, agent: AgentProfile):
        self.agents.append(agent)
        self._add_agent(agent)

    def get_agent(self, name: Optional[str] = None, role: Optional[str] = None) -> Optional[AgentProfile]:
        if name:
            return self._by_name.get(name)
        if role:
            agents = self._by_role.get(role, [])
            return agents[0] if agents else None
        return None

    def get_agents_by_role(self, role: str) -> List[AgentProfile]:
        return self._by_role.get(role, [])

    def __len__(self) -> int:
        return len(self.agents)

    def __iter__(self):
        return iter(self.agents)

    def __repr__(self) -> str:
        return f"AgentTeam(agents={len(self.agents)}, roles={list(self._by_role.keys())})"

    def validate(self) -> List[str]:
        errors = []
        if not self.agents:
            errors.append("Team has no agents")
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            errors.append("Duplicate agent names found")
        return errors


def load_team_from_yaml(path: str) -> AgentTeam:
    """Load agent team from YAML file."""
    log = logging.getLogger(__name__)
    log.info(f"Loading agent team from: {path}")

    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)

    agents = [AgentProfile.from_dict(d) for d in data.get("agents", [])]
    team = AgentTeam(agents)
    log.info(f"Loaded team with {len(team)} agents")
    return team


def load_team_from_dict(data: Dict[str, Any]) -> AgentTeam:
    """Load agent team from dictionary."""
    return AgentTeam([AgentProfile.from_dict(d) for d in data.get("agents", [])])


def save_team_to_yaml(team: AgentTeam, path: str):
    """Save agent team to YAML file."""
    log = logging.getLogger(__name__)
    log.info(f"Saving agent team to: {path}")
    data = {"agents": [agent.to_dict() for agent in team.agents]}
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    log.info(f"Saved {len(team)} agents")


# Pre-defined team templates
DEFAULT_TEAM_TEMPLATES = {
    "dev_team": {
        "agents": [
            {
                "name": "TechLead",
                "role": "technical_lead",
                "system_prompt": "You are a technical lead responsible for architectural decisions and code reviews.",
                "instance_prompt": "Review this technical design: {problem_statement}",
                "description": "Technical Lead - makes architecture decisions",
                "capabilities": ["architecture", "code_review", " mentoring"],
            },
            {
                "name": "SeniorDev",
                "role": "developer",
                "system_prompt": "You are a senior software engineer specializing in clean, maintainable code.",
                "instance_prompt": "Implement: {problem_statement}",
                "description": "Senior Developer - implements features",
                "capabilities": ["implementation", "testing", "debugging"],
            },
            {
                "name": "JuniorDev",
                "role": "developer",
                "system_prompt": "You are a junior developer learning best practices. Ask clarifying questions when needed.",
                "instance_prompt": "Task: {problem_statement}",
                "description": "Junior Developer - learns and implements simple tasks",
                "capabilities": ["implementation", "learning"],
            },
        ]
    },
    "product_team": {
        "agents": [
            {
                "name": "ProductManager",
                "role": "product_manager",
                "system_prompt": "You are a product manager focused on user needs and business value.",
                "instance_prompt": "Define requirements for: {problem_statement}",
                "description": "Product Manager - defines requirements",
                "capabilities": ["requirements", "prioritization", "user_research"],
            },
            {
                "name": "Designer",
                "role": "designer",
                "system_prompt": "You are a UX designer focused on user-friendly interfaces.",
                "instance_prompt": "Design UI for: {problem_statement}",
                "description": "Designer - creates UI/UX",
                "capabilities": ["UI_design", "UX_research", "prototyping"],
            },
            {
                "name": "Developer",
                "role": "developer",
                "system_prompt": "You are a full-stack developer.",
                "instance_prompt": "Implement: {problem_statement}",
                "description": "Developer - builds features",
                "capabilities": ["frontend", "backend", "database"],
            },
        ]
    },
}


def create_team_from_template(template_name: str) -> AgentTeam:
    """Create a pre-defined team from template."""
    log = logging.getLogger(__name__)
    log.info(f"Creating team from template: {template_name}")

    if template_name not in DEFAULT_TEAM_TEMPLATES:
        available = list(DEFAULT_TEAM_TEMPLATES.keys())
        log.error(f"Unknown template: {template_name}, available: {available}")
        raise ValueError(f"Unknown template: {template_name}. Available: {available}")

    team = load_team_from_dict(DEFAULT_TEAM_TEMPLATES[template_name])
    log.info(f"Created team: {team}")
    return team


def profile_to_agent_args(
    profile: AgentProfile,
    llm_name: str = "openai/gpt-4",
    llm_base_url: Optional[str] = None,
    max_retries: int = 5,
    **kwargs,
) -> Any:
    """Convert AgentProfile to AgentArgs for use with Agent."""
    log = logging.getLogger(__name__)
    log.debug(f"Converting profile '{profile.name}' to AgentArgs (llm={llm_name})")

    args = AgentArgs(
        system_prompt=profile.system_prompt,
        instance_prompt=profile.instance_prompt,
        llm_name=llm_name,
        llm_base_url=llm_base_url,
        max_retries=max_retries,
        **kwargs,
    )

    log.debug(f"Profile '{profile.name}' -> AgentArgs: system_prompt length={len(args.system_prompt)}")
    return args


def run_agent_team(
    team: AgentTeam,
    task: str,
    llm_name: str = "openai/gpt-4",
    runtime=None,
    llm_base_url: Optional[str] = None,
    max_steps: int = 30,
) -> Dict[str, Any]:
    """Run a task across an agent team - each agent processes independently."""
    log = logging.getLogger(__name__)
    log.info("=" * 50)
    log.info(f"Starting team execution: {len(team)} agents")
    log.info(f"Task: {task[:100]}...")
    log.info(f"LLM: {llm_name}, max_steps: {max_steps}")
    log.info("=" * 50)

    results = {}

    for profile in team:
        log.info(f"[{profile.name}] Starting task execution...")

        # Create AgentArgs from profile
        args = profile_to_agent_args(
            profile,
            llm_name=llm_name,
            llm_base_url=llm_base_url,
        )

        # Create agent
        agent = Agent(args)

        # Run agent (runtime first, then problem_statement)
        try:
            log.debug(f"[{profile.name}] Running agent...")
            result = agent.run(runtime, task, max_steps=max_steps)
            results[profile.name] = {
                "role": profile.role,
                "success": True,
                "result": result,
            }
            log.info(f"[{profile.name}] ✓ Completed successfully")
        except Exception as e:
            log.error(f"[{profile.name}] ✗ Failed: {e}")
            results[profile.name] = {
                "role": profile.role,
                "success": False,
                "error": str(e),
            }

    # Summary
    success_count = sum(1 for r in results.values() if r.get("success"))
    log.info("=" * 50)
    log.info(f"Team execution complete: {success_count}/{len(team)} succeeded")
    log.info("=" * 50)

    return results

__all__ = [
    "AgentProfile",
    "AgentTeam",
    "load_team_from_yaml",
    "save_team_to_yaml",
    "create_team_from_template",
    "DEFAULT_TEAM_TEMPLATES",
    "profile_to_agent_args",
    "run_agent_team",
]
