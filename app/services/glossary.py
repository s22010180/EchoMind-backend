"""
Software-engineering vocabulary used to show WHICH technical terms an answer contained.

This is evidence for the examiner ("the student mentioned: polymorphism, interface, REST"), never a
correctness judgement - a term can be used wrongly. Edit SE_TERMS freely to match your syllabus.
"""
from __future__ import annotations

import re
from typing import List, Set

SE_TERMS: Set[str] = {
    # context-bearing replacements for ambiguous everyday words (e.g. "the rest of the system" must not match)
    "rest api", "rest service", "rest endpoint", "restful", "rest architecture", "node js", "nodejs", "express js",
    "expressjs", "git branch", "branching strategy", "merge conflict", "sql join", "inner join", "outer join",
    "left join", "thread safe", "thread pool", "docker container", "containerization", "container orchestration",
    "cloud computing", "cloud native", "cloud service", "factory pattern", "factory method", "adapter pattern",
    "proxy pattern", "observer pattern", "strategy pattern", "builder pattern", "decorator pattern",
    "iterator pattern", "prototype pattern", "prototyping", "mock object", "test double", "database transaction",
    "effort estimation", "cost estimation", "requirements engineering", "requirements elicitation",
    "ci pipeline", "build pipeline", "deployment pipeline", "data pipeline", "caching layer", "redis",
    # object orientation & design
    "object oriented", "inheritance", "polymorphism", "encapsulation", "abstraction", "abstract class",
    "constructor", "overriding", "overloading", "composition", "aggregation", "coupling", "cohesion",
    "solid principles", "single responsibility", "open closed", "liskov", "dependency inversion",
    "dependency injection", "design pattern", "singleton", "abstract factory",     "facade", "command pattern",
    "template method", "mvc", "mvvm", "mvp", "repository pattern", "separation of concerns",
    "modularity", "interface segregation",
    # architecture & systems
    "microservice", "monolith", "monolithic", "service oriented", "restful", "api", "graphql", "soap",
    "grpc", "websocket", "webrtc", "event driven", "message queue", "message broker", "kafka", "rabbitmq",
    "publish subscribe", "pub sub", "load balancer", "load balancing", "caching", "cdn",
    "reverse proxy", "api gateway", "serverless", "docker", "kubernetes", "virtual machine",
    "aws", "azure", "scalability", "scalable", "horizontal scaling", "vertical scaling",
    "high availability", "fault tolerance", "latency", "throughput", "concurrency", "multithreading",
    "race condition", "deadlock", "asynchronous", "synchronous", "client server", "three tier",
    "layered architecture", "clean architecture", "hexagonal architecture", "domain driven", "middleware",
    "orm", "idempotent", "stateless",
    # process & project
    "software development life cycle", "sdlc", "waterfall", "agile", "scrum", "kanban", "extreme programming",
    "sprint", "product backlog", "sprint backlog", "user story", "story points", "stand up", "retrospective",
    "product owner", "scrum master", "stakeholder", "functional requirement",
    "non functional requirement", "use case", "uml", "class diagram", "sequence diagram", "activity diagram",
    "er diagram", "entity relationship", "data flow diagram", "feasibility", "risk management",
    "gantt", "minimum viable product", "mvp", "iterative", "incremental", "spiral model",
    "v model", "devops", "ci/cd", "ci cd", "continuous integration", "continuous delivery",
    "continuous deployment", "version control", "git", "github",     "pull request", "code review", "refactoring", "technical debt", "code smell", "maintainability",
    "reusability", "traceability", "rollback",
    # testing & quality
    "unit test", "unit testing", "integration test", "integration testing", "system testing",
    "acceptance testing", "regression testing", "smoke test", "test case", "test plan",
    "test driven development", "tdd", "bdd", "mocking", "test coverage", "code coverage",
    "black box", "white box", "boundary value", "equivalence partitioning", "load testing", "stress testing",
    "performance testing", "usability testing", "selenium", "jest", "junit", "pytest", "debugging",
    "defect", "static analysis",
    # data
    "database", "sql", "nosql", "mongodb", "postgresql", "mysql", "schema", "primary key", "foreign key",
    "normalization", "normalisation", "denormalization", "indexing", "acid",     "stored procedure", "migration", "relational", "cardinality",
    # security
    "authentication", "authorization", "authorisation", "encryption", "hashing", "salting", "jwt",
    "json web token", "oauth", "https", "ssl", "tls", "sql injection", "xss", "cross site scripting", "csrf",
    "firewall", "vulnerability", "penetration testing", "role based access", "access control",
    "two factor", "multi factor", "cors",
    # programming & web
    "algorithm", "data structure", "linked list", "hash table", "binary tree", "recursion",
    "time complexity", "space complexity", "big o", "frontend", "backend", "full stack", "react", "angular",
    "vue", "django", "flask", "fastapi", "spring boot", "typescript", "javascript",
    "python", "java", "html", "css", "responsive design", "state management", "redux", "virtual dom",
    "compiler", "interpreter", "exception handling",     
    # AI / data science
    "machine learning", "deep learning", "neural network", "training data", "dataset", "overfitting",
    "underfitting", "precision", "recall", "f1 score", "confusion matrix", "cross validation",
    "classification", "clustering", "transformer", "speech recognition", "natural language processing",
}


def _build_pattern() -> "re.Pattern[str]":
    parts = []
    for term in sorted(SE_TERMS, key=len, reverse=True):
        body = r"[\s\-]+".join(re.escape(w) for w in term.split())
        parts.append(body)
    # optional plural, not glued to other letters/digits
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?:s|es)?(?![A-Za-z0-9])", re.IGNORECASE)


_PATTERN = _build_pattern()


def _canonical(raw: str) -> str:
    norm = re.sub(r"[\s\-]+", " ", raw.lower()).strip()
    if norm in SE_TERMS:
        return norm
    for cut in (2, 1):
        if len(norm) > cut and norm[:-cut] in SE_TERMS:
            return norm[:-cut]
    return norm


def find_terms(text: str) -> List[str]:
    """Distinct technical terms found in `text`, canonical lowercase form, alphabetical."""
    if not text:
        return []
    return sorted({_canonical(m.group(0)) for m in _PATTERN.finditer(text)})


# Words Whisper tends to mis-hear in software-engineering speech. They are passed as a vocabulary hint
# ("initial prompt") so the recogniser prefers these spellings. Keep the list short (the prompt has a token limit).
ASR_HINT_TERMS = (
    "polymorphism", "inheritance", "encapsulation", "abstraction", "microservices", "monolith", "REST API",
    "GraphQL", "WebSocket", "Docker", "Kubernetes", "CI/CD", "Git", "pull request", "unit testing",
    "integration testing", "TDD", "Agile", "Scrum", "UML", "sequence diagram", "database normalization", "SQL",
    "NoSQL", "PostgreSQL", "authentication", "authorization", "JWT", "OAuth", "dependency injection",
    "design pattern", "singleton", "observer pattern", "MVC", "refactoring", "technical debt", "scalability",
    "latency", "load balancer", "React", "FastAPI", "Python", "TypeScript",
)


def asr_prompt() -> str:
    return "Software engineering viva. Technical terms: " + ", ".join(ASR_HINT_TERMS) + "."
