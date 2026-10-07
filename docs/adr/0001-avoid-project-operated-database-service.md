# Avoid a project-operated database service

GradLab will minimize separately operated services and will not require a project-operated relational database service. Current workflows may use an embedded file-backed store such as SQLite because simpler credential-free operation and recovery are more valuable at the present scale than the coordination and scaling features of a managed database service; this decision belongs to the orchestration architecture and is not a permanent product requirement.
